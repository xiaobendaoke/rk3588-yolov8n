/**
 * rknn_infer.h - C API for RKNN YOLO11 inference
 *
 * 供 Python ctypes 调用的 C 接口
 */

#ifndef RKNN_INFER_H
#define RKNN_INFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECTIONS 128
#define NUM_CLASSES 80

typedef struct {
    int class_id;
    float confidence;
    int x1, y1, x2, y2;
} Detection;

typedef struct {
    Detection dets[MAX_DETECTIONS];
    int count;
} DetectionResult;

/**
 * 创建推理引擎
 * @return 引擎句柄，失败返回 NULL
 */
void* rknn_engine_create(const char* model_path, int input_size,
                          float conf_threshold, float nms_threshold);

/**
 * 对图像帧执行推理
 * @param engine 引擎句柄
 * @param img_data 图像数据 (BGR, HWC, uint8)
 * @param img_width 图像宽度
 * @param img_height 图像高度
 * @param result 输出检测结果
 * @return 0 成功，非 0 失败
 */
int rknn_engine_infer(void* engine, const uint8_t* img_data,
                       int img_width, int img_height,
                       DetectionResult* result);

/**
 * 销毁推理引擎
 */
void rknn_engine_destroy(void* engine);

#ifdef __cplusplus
}
#endif

#endif // RKNN_INFER_H
