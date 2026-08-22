#ifndef MODEL_YOLO26_H
#define MODEL_YOLO26_H

#include "module_vipnn.h"

extern nnmodel_t yolo26;

/* fills buf with decode state: layout, tensor count, nc, cls-is-prob, input width */
void yolo26_get_debug_info(char *buf, int len);

#endif
