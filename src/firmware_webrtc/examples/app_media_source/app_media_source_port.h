/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef APP_MEDIA_SOURCE_PORT_H
#define APP_MEDIA_SOURCE_PORT_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "transceiver_data_types.h"

typedef struct MediaFrame {
    uint8_t * pData;
    uint32_t size;
    uint64_t timestampUs;
    TransceiverTrackKind_t trackKind;
    uint8_t freeData;  /* indicate user need to free pData after using it */
} MediaFrame_t;

typedef int32_t (* OnFrameReadyToSend_t)( void * pCtx,
                                          MediaFrame_t * pFrame );

/* Detection metadata produced on-device by the NN, handed to the app so it can
 * be sent to viewers over the WebRTC data channel.
 *
 * The NN results are otherwise burned into the video as OSD rectangles and
 * printed to the serial console -- neither of which a remote consumer can
 * read as numbers. `pJson` is a NUL-terminated UTF-8 JSON object owned by the
 * caller and only valid for the duration of the call: copy it if you need to
 * keep it. */
typedef int32_t (* OnMetadataReadyToSend_t)( void * pCtx,
                                             const char * pJson,
                                             uint32_t length );

void AppMediaSourcePort_RegisterMetadataSink( OnMetadataReadyToSend_t onMetadataReadyToSendFunc,
                                              void * pOnMetadataReadyToSendCustomContext );

int32_t AppMediaSourcePort_Init( void );
int32_t AppMediaSourcePort_Start( OnFrameReadyToSend_t onVideoFrameReadyToSendFunc,
                                  void * pOnVideoFrameReadyToSendCustomContext,
                                  OnFrameReadyToSend_t onAudioFrameReadyToSendFunc,
                                  void * pOnAudioFrameReadyToSendCustomContext );
void AppMediaSourcePort_Stop( void );
void AppMediaSourcePort_Destroy( void );
void AppMediaSourcePort_PlayAudioFrame( MediaFrame_t * pFrame );

#ifdef __cplusplus
}
#endif

#endif /* APP_MEDIA_SOURCE_PORT_H */
