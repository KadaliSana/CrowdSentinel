/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2012, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2012, by Michael Tuexen. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a) Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * b) Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
 *
 * c) Neither the name of Cisco Systems, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/sys/netinet/sctp_timer.h 359195 2020-03-21 16:12:19Z tuexen $");
#endif

#ifndef _NETINET_USER_CONFIG_H_
#define _NETINET_USER_CONFIG_H_

#define SCTP_THREAD_DEFAULT_NAME "pthread"
#define SCTP_THREAD_DEFAULT_SIZE 4096
#define SCTP_THREAD_ADDR_MON_NAME "SCTP addr mon"
#define SCTP_THREAD_ADDR_MON_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_IPV4_RCV_NAME "SCTP/IP4"
#define SCTP_THREAD_IPV4_RCV_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_UDP_IPV4_RCV_NAME "SCTP/UDP/IP4"
#define SCTP_THREAD_UDP_IPV4_RCV_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_IPV6_RCV_NAME "SCTP/IP6"
#define SCTP_THREAD_IPV6_RCV_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_IPV6_RCV_NAME "SCTP/IP6"
#define SCTP_THREAD_IPV6_RCV_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_UDP_IPV6_RCV_NAME "SCTP/UDP/IP6"
#define SCTP_THREAD_UDP_IPV6_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_ITERATOR_NAME "SCTP iterator"
#define SCTP_THREAD_ITERATOR_SIZE SCTP_THREAD_DEFAULT_SIZE
#define SCTP_THREAD_TIMER_NAME "SCTP timer"
#define SCTP_THREAD_TIMER_SIZE 8192//SCTP_THREAD_DEFAULT_SIZE

#endif//!< _NETINET_USER_CONFIG_H_
