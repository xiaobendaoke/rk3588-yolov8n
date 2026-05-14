/**
 * rknn_infer.cpp - RKNN YOLO11 推理库 (多线程版)
 *
 * 支持多 NPU 核心并行推理
 */

#include "rknn_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <set>
#include <atomic>

#include "rknn_api.h"

// ============ 工具函数 ============

static inline int clamp_i(int val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

static float calc_iou(float x1_0, float y1_0, float x2_0, float y2_0,
                      float x1_1, float y1_1, float x2_1, float y2_1) {
    float w = fmaxf(0.f, fminf(x2_0, x2_1) - fmaxf(x1_0, x1_1) + 1.0f);
    float h = fmaxf(0.f, fminf(y2_0, y2_1) - fmaxf(y1_0, y1_1) + 1.0f);
    float inter = w * h;
    float area0 = (x2_0 - x1_0 + 1.0f) * (y2_0 - y1_0 + 1.0f);
    float area1 = (x2_1 - x1_1 + 1.0f) * (y2_1 - y1_1 + 1.0f);
    float u = area0 + area1 - inter;
    return u <= 0.f ? 0.f : (inter / u);
}

static void compute_dfl(const float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[16];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

// ============ 引擎结构 ============

struct Worker {
    rknn_context ctx;
    int n_inputs;
    int n_outputs;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_output* outputs;
    bool is_quant;
    bool is_fp16;
};

struct RknnEngine {
    int input_size;
    float conf_threshold;
    float nms_threshold;
    std::vector<Worker> workers;
    std::atomic<int> next_idx{0};
};

static void* load_model_file(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* data = malloc(*size);
    if (data) {
        size_t read = fread(data, 1, *size, fp);
        (void)read;
    }
    fclose(fp);
    return data;
}

static int init_worker(Worker* w, unsigned char* model_data, int model_size, int core_id) {
    // 设置核心掩码
    int ret = rknn_init(&w->ctx, model_data, model_size, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init failed for core %d: %d\n", core_id, ret);
        return ret;
    }

    // 设置 NPU 核心掩码 (RK3588: core0=1, core1=2, core2=4)
    int core_mask = 1 << core_id;
    ret = rknn_set_core_mask(w->ctx, (rknn_core_mask)core_mask);
    if (ret < 0) {
        fprintf(stderr, "rknn_set_core_mask failed for core %d: %d (continuing with auto)\n", core_id, ret);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(w->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) return ret;

    w->n_inputs = io_num.n_input;
    w->n_outputs = io_num.n_output;

    w->input_attrs = new rknn_tensor_attr[w->n_inputs];
    for (int i = 0; i < w->n_inputs; i++) {
        w->input_attrs[i].index = i;
        rknn_query(w->ctx, RKNN_QUERY_INPUT_ATTR, &w->input_attrs[i], sizeof(rknn_tensor_attr));
    }

    w->output_attrs = new rknn_tensor_attr[w->n_outputs];
    for (int i = 0; i < w->n_outputs; i++) {
        w->output_attrs[i].index = i;
        rknn_query(w->ctx, RKNN_QUERY_OUTPUT_ATTR, &w->output_attrs[i], sizeof(rknn_tensor_attr));
    }

    w->is_quant = (w->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                   w->output_attrs[0].type == RKNN_TENSOR_INT8);
    w->is_fp16 = (w->output_attrs[0].type == RKNN_TENSOR_FLOAT16 ||
                  w->output_attrs[0].type == RKNN_TENSOR_FLOAT32);

    w->outputs = new rknn_output[w->n_outputs];
    memset(w->outputs, 0, sizeof(rknn_output) * w->n_outputs);

    return 0;
}

void* rknn_engine_create(const char* model_path, int input_size,
                          float conf_threshold, float nms_threshold) {
    RknnEngine* eng = new RknnEngine();
    eng->input_size = input_size;
    eng->conf_threshold = conf_threshold;
    eng->nms_threshold = nms_threshold;

    int model_size = 0;
    unsigned char* model_data = (unsigned char*)load_model_file(model_path, &model_size);
    if (!model_data) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        delete eng;
        return NULL;
    }

    // 创建 3 个 worker，绑定到 3 个 NPU 核心
    int n_workers = 3;
    eng->workers.resize(n_workers);

    for (int i = 0; i < n_workers; i++) {
        int ret = init_worker(&eng->workers[i], model_data, model_size, i);
        if (ret < 0) {
            fprintf(stderr, "Failed to init worker %d\n", i);
            // 回退到单 worker
            eng->workers.resize(i > 0 ? i : 1);
            break;
        }
    }

    free(model_data);

    printf("RKNN engine created: %s\n", model_path);
    printf("  workers: %d, input_size: %d\n", (int)eng->workers.size(), input_size);
    printf("  fp16: %s\n", eng->workers[0].is_fp16 ? "yes" : "no");

    return eng;
}

int rknn_engine_infer(void* engine, const uint8_t* img_data,
                       int img_width, int img_height,
                       DetectionResult* result) {
    if (!engine || !img_data || !result) return -1;

    RknnEngine* eng = (RknnEngine*)engine;
    result->count = 0;

    // 选择 worker (轮询)
    int idx = eng->next_idx.fetch_add(1) % eng->workers.size();
    Worker* w = &eng->workers[idx];

    int in_size = eng->input_size;

    // 设置输入 (已经是 resize 后的 640x640 RGB)
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = in_size * in_size * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = (void*)img_data;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(w->ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(w->ctx, NULL);
    if (ret < 0) return -1;

    for (int i = 0; i < w->n_outputs; i++) {
        w->outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(w->ctx, w->n_outputs, w->outputs, NULL);
    if (ret < 0) return -1;

    // 后处理
    std::vector<float> filter_boxes;
    std::vector<float> obj_probs;
    std::vector<int> class_ids;

    int output_per_branch = w->n_outputs / 3;

    for (int branch = 0; branch < 3; branch++) {
        int box_idx = branch * output_per_branch;
        int score_idx = branch * output_per_branch + 1;

        int dfl_len = w->output_attrs[box_idx].dims[1] / 4;
        int grid_h = w->output_attrs[box_idx].dims[2];
        int grid_w = w->output_attrs[box_idx].dims[3];
        int stride = in_size / grid_h;
        int grid_len = grid_h * grid_w;

        float* box_data = (float*)w->outputs[box_idx].buf;
        float* score_data = (float*)w->outputs[score_idx].buf;
        float* score_sum = NULL;
        if (output_per_branch == 3) {
            score_sum = (float*)w->outputs[branch * output_per_branch + 2].buf;
        }

        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int offset = i * grid_w + j;
                int max_class_id = -1;

                if (score_sum && score_sum[offset] < eng->conf_threshold)
                    continue;

                float max_score = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float s = score_data[offset];
                    if (s > eng->conf_threshold && s > max_score) {
                        max_score = s;
                        max_class_id = c;
                    }
                    offset += grid_len;
                }

                if (max_score > eng->conf_threshold) {
                    offset = i * grid_w + j;
                    float box[4];
                    float before_dfl[64]; // dfl_len*4 max
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = box_data[offset];
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + j + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;

                    filter_boxes.push_back(x1);
                    filter_boxes.push_back(y1);
                    filter_boxes.push_back(x2 - x1);
                    filter_boxes.push_back(y2 - y1);
                    obj_probs.push_back(max_score);
                    class_ids.push_back(max_class_id);
                }
            }
        }
    }

    rknn_outputs_release(w->ctx, w->n_outputs, w->outputs);

    // NMS
    int valid_count = (int)obj_probs.size();
    if (valid_count == 0) return 0;

    // 排序
    std::vector<int> sorted_idx(valid_count);
    for (int i = 0; i < valid_count; i++) sorted_idx[i] = i;
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        return obj_probs[a] > obj_probs[b];
    });

    std::vector<bool> suppressed(valid_count, false);
    int out_count = 0;

    for (int i = 0; i < valid_count && out_count < MAX_DETECTIONS; i++) {
        int idx = sorted_idx[i];
        if (suppressed[idx]) continue;

        float x1 = filter_boxes[idx * 4 + 0];
        float y1 = filter_boxes[idx * 4 + 1];
        float w_val = filter_boxes[idx * 4 + 2];
        float h_val = filter_boxes[idx * 4 + 3];

        result->dets[out_count].class_id = class_ids[idx];
        result->dets[out_count].confidence = obj_probs[idx];
        result->dets[out_count].x1 = clamp_i((int)x1, 0, in_size - 1);
        result->dets[out_count].y1 = clamp_i((int)y1, 0, in_size - 1);
        result->dets[out_count].x2 = clamp_i((int)(x1 + w_val), 0, in_size - 1);
        result->dets[out_count].y2 = clamp_i((int)(y1 + h_val), 0, in_size - 1);
        out_count++;

        for (int j = i + 1; j < valid_count; j++) {
            int jdx = sorted_idx[j];
            if (suppressed[jdx] || class_ids[jdx] != class_ids[idx]) continue;

            float jx1 = filter_boxes[jdx * 4 + 0];
            float jy1 = filter_boxes[jdx * 4 + 1];
            float jw = filter_boxes[jdx * 4 + 2];
            float jh = filter_boxes[jdx * 4 + 3];

            if (calc_iou(x1, y1, x1 + w_val, y1 + h_val, jx1, jy1, jx1 + jw, jy1 + jh) > eng->nms_threshold)
                suppressed[jdx] = true;
        }
    }

    result->count = out_count;
    return 0;
}

void rknn_engine_destroy(void* engine) {
    if (!engine) return;
    RknnEngine* eng = (RknnEngine*)engine;
    for (auto& w : eng->workers) {
        delete[] w.outputs;
        delete[] w.input_attrs;
        delete[] w.output_attrs;
        rknn_destroy(w.ctx);
    }
    delete eng;
}
