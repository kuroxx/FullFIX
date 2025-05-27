/*
Copyright (c) 2015, Maxim Konakov
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _GNU_SOURCE

// Compiler-specific optimization pragmas
#ifdef __GNUC__
#pragma GCC optimize ("O3")
#pragma GCC target("sse4.2")
#endif

#ifdef _MSC_VER
#pragma optimize("gt", on)
#endif

#include "fix_impl.h"

#ifdef USE_SSE
#include <xmmintrin.h>
#include <smmintrin.h> // For SSE4.1 functions
#endif

// Common branch prediction macros
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)      __builtin_expect(!!(x), 1)
#define UNLIKELY(x)    __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)      (x)
#define UNLIKELY(x)    (x)
#endif

// Alignment macros
#if defined(__GNUC__) || defined(__clang__)
#define ALIGNED(n)     __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define ALIGNED(n)     __declspec(align(n))
#else
#define ALIGNED(n)
#endif

// Prefetch hint
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr)
#endif

// Growth factor for reallocation to reduce frequency
#define GROWTH_FACTOR 1.8

// message buffer handling - optimized with better prefetching
static
char* make_space(fix_parser* const restrict parser, char* dest, unsigned extra_len)
{
	const unsigned n = dest - parser->body, len = n + extra_len;

	if(UNLIKELY(len > parser->body_capacity))
	{
		// reallocate memory with improved growth factor
		unsigned new_capacity = parser->body_capacity * GROWTH_FACTOR;
		if(new_capacity < len) new_capacity = len;
        
        // Align to cache line boundary for better memory access
        new_capacity = (new_capacity + 63) & ~63;
        
		// Use aligned allocation if available
		#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
			void* p = NULL;
			if(posix_memalign(&p, 64, new_capacity) != 0) {
				set_fatal_error(parser, FE_OUT_OF_MEMORY);
				return NULL;
			}
			
			// Copy data and free old buffer
			memcpy(p, parser->body, n);
			free(parser->body);
			char* new_ptr = (char*)p;
		#else
			char* const new_ptr = realloc(parser->body, new_capacity);
			if(UNLIKELY(!new_ptr)) {
				set_fatal_error(parser, FE_OUT_OF_MEMORY);
				return NULL;
			}
		#endif

		// recalculate dest pointer
		dest = new_ptr + n;

		// recalculate context
		parser->result.error.context = (fix_string){ new_ptr + (parser->result.error.context.begin - parser->body), dest };

		// store new pointer
		parser->body = new_ptr;
		parser->body_capacity = new_capacity;
	}
	
	// Prefetch to reduce cache misses
	PREFETCH(dest);

	return dest;
}

// initialisation - optimized with aligned memory allocation
bool init_scanner(fix_parser* restrict parser)
{
    // Use posix_memalign for better aligned memory if available
    #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        void* p = NULL;
        // Align to 64-byte boundary (typical cache line size) for better cache efficiency
        if(posix_memalign(&p, 64, INITIAL_BODY_SIZE) != 0)
            return false;
        parser->body = p;
    #else
        // If posix_memalign isn't available, use standard malloc
        char* const p = malloc(INITIAL_BODY_SIZE);
        if(UNLIKELY(!p))
            return false;
        parser->body = p;
    #endif

    parser->body_capacity = INITIAL_BODY_SIZE;
    return true;
}

// scanner helper functions
static inline
unsigned min(unsigned a, unsigned b)
{
	return a < b ? a : b;
}

// copy a chunk of 'state->counter' bytes
static
bool copy_chunk(scanner_state* const restrict state)
{
	const char* const s = state->src;
	const unsigned n = min(state->end - s, state->counter);

	// Prefetch next data chunk
	PREFETCH(s + 64);
    
	state->dest = mempcpy(state->dest, s, n);
	state->src = s + n;
	state->counter -= n;
	return LIKELY(state->counter == 0);
}

// Optimized checksum calculation using vectorized operations where possible
static
unsigned char copy_cs(char* restrict p, const char* restrict s, unsigned n)
{
	unsigned char cs = 0;
	const char* const end = s + n;

#ifdef USE_SSE
	if(LIKELY(end - s >= 16))
	{
		__m128i cs128 = _mm_setzero_si128();

		// Process 16-byte chunks with aligned loads where possible
		if(((uintptr_t)s & 15) == 0 && ((uintptr_t)p & 15) == 0) {
			// Both source and destination are aligned
			for(; LIKELY(end - s >= 16); s += 16, p += 16) {
				PREFETCH(s + 128);  // Prefetch further ahead
				
				const __m128i tmp = _mm_load_si128((const __m128i*)s);
				_mm_store_si128((__m128i*)p, tmp);
				cs128 = _mm_add_epi8(cs128, tmp);
			}
		} else {
			// Process with unaligned loads
			for(; LIKELY(end - s >= 16); s += 16, p += 16) {
				PREFETCH(s + 128);  // Prefetch further ahead
				
				const __m128i tmp = _mm_loadu_si128((const __m128i*)s);
				_mm_storeu_si128((__m128i*)p, tmp);
				cs128 = _mm_add_epi8(cs128, tmp);
			}
		}

		// Horizontal sum of bytes
		cs128 = _mm_add_epi8(cs128, _mm_srli_si128(cs128, 8));
		cs128 = _mm_add_epi8(cs128, _mm_srli_si128(cs128, 4));
		cs128 = _mm_add_epi8(cs128, _mm_srli_si128(cs128, 2));
		cs128 = _mm_add_epi8(cs128, _mm_srli_si128(cs128, 1));
        
		#ifdef __SSE4_1__
			cs += _mm_extract_epi8(cs128, 0);
		#else
			cs += _mm_extract_epi16(cs128, 0);
		#endif
	}

	// Process 8-byte remainder
	if(LIKELY(end - s >= 8))
	{
		__m128i cs64 = _mm_loadl_epi64((const __m128i*)s);
		_mm_storel_epi64((__m128i*)p, cs64);
		s += 8;
		p += 8;
		
		cs64 = _mm_add_epi8(cs64, _mm_srli_si128(cs64, 4));
		cs64 = _mm_add_epi8(cs64, _mm_srli_si128(cs64, 2));
		cs64 = _mm_add_epi8(cs64, _mm_srli_si128(cs64, 1));
        
		#ifdef __SSE4_1__
			cs += _mm_extract_epi8(cs64, 0);
		#else
			cs += _mm_extract_epi16(cs64, 0);
		#endif
	}
#endif	// #ifdef USE_SSE

	// Process remaining bytes with unrolled loop for small gains
	if(end - s >= 4) {
		cs += (*p++ = *s++);
		cs += (*p++ = *s++);
		cs += (*p++ = *s++);
		cs += (*p++ = *s++);
	}
	
	while(s < end)
		cs += (*p++ = *s++);

	return cs;
}

static
bool copy_chunk_cs(scanner_state* const restrict state)
{
	const unsigned n = min(state->end - state->src, state->counter);

	// Prefetch next data chunk
	PREFETCH(state->src + 128);
    
	state->check_sum += copy_cs(state->dest, state->src, n);
	state->src += n;
	state->dest += n;
	state->counter -= n;
	return LIKELY(state->counter == 0);
}

// covert and validate message length - optimized with better branch prediction
static
bool convert_message_length(scanner_state* const restrict state)
{
	if(UNLIKELY(state->counter < 2))
		return false;

	const char* s = state->dest - state->counter;
	unsigned len = CHAR_TO_INT(*s++) - '0';

	if(UNLIKELY(len > 9))
		return false;

	const char* const end = state->dest - 1;

	// Use specific constant limits for better branch prediction
	unsigned digits = 1;
	
	while(s < end)
	{
		const unsigned t = CHAR_TO_INT(*s++) - '0';

		if(UNLIKELY(t > 9))
			return false;

		len = len * 10 + t;
		digits++;
		
		// Early termination if we've exceeded max length
		if(UNLIKELY(len > MAX_MESSAGE_LENGTH))
			return false;
	}

	if(UNLIKELY(len < sizeof("35=0|49=X|56=Y|34=1|") - 1))
		return false;

	state->counter = len;
	return true;
}

static
bool valid_checksum(const scanner_state* const restrict state)
{
	const unsigned
		cs2 = CHAR_TO_INT(state->dest[-4]) - '0',
		cs1 = CHAR_TO_INT(state->dest[-3]) - '0',
		cs0 = CHAR_TO_INT(state->dest[-2]) - '0';

	return cs2 <= 9 && cs1 <= 9 && cs0 <= 9 && (unsigned)state->check_sum == cs2 * 100 + cs1 * 10 + cs0;
}

// Scanner - optimized with improved state machine flow and hash-based message deduplication
bool extract_next_message(fix_parser* const restrict parser)
{
	scanner_state* const state = &parser->state;

	// Use computed goto when available for faster state machine
	#if defined(__GNUC__) || defined(__clang__)
	static const void* const labels[] = {
		&&INIT_STATE,
		&&MESSAGE_HEADER_STATE,
		&&MESSAGE_LENGTH_STATE,
		&&MESSAGE_BODY_STATE,
		&&TRAILER_STATE
	};
	
	goto *labels[state->label];
	#else
	switch(state->label)
	{
	#endif

	#if defined(__GNUC__) || defined(__clang__)
	INIT_STATE:
	#else
	case 0:	// initialisation
	#endif
		// clear previous error
		parser->result.error = (fix_error_details){ FE_OK, 0, (fix_string){ parser->body, NULL }, EMPTY_STR };
		parser->result.msg_type_code = -1;

		// make new state
		state->dest = parser->body;
		state->counter = parser->header_len;
		
		// Prefetch header data
		PREFETCH(parser->header);
		
		// Continue to next state
		#if defined(__GNUC__) || defined(__clang__)
		goto MESSAGE_HEADER_STATE;
		#else
		// fall through
	#endif

	#if defined(__GNUC__) || defined(__clang__)
	MESSAGE_HEADER_STATE:
	#else
	case 1:	// message header
	#endif
		// copy header
		if(state->src == state->end || !copy_chunk(state)) {
			state->label = 1;
			return false;
		}

		// validate header using optimized compare
		if(UNLIKELY(memcmp(parser->header, parser->body, parser->header_len) != 0))
			goto BEGIN_STRING_FAILURE;

		state->check_sum = parser->header_checksum;
		state->counter = 0;

		// update context
		parser->result.error.context.begin = state->dest;
		
		// Continue to next state
		#if defined(__GNUC__) || defined(__clang__)
		goto MESSAGE_LENGTH_STATE;
		#else
		// fall through
	#endif

	#if defined(__GNUC__) || defined(__clang__)
	MESSAGE_LENGTH_STATE:
	#else
	case 2:	// message length
	#endif
		// copy bytes
		for(;;)
		{
			if(UNLIKELY(state->src == state->end)) {
				state->label = 2;
				return false;
			}

			const unsigned char x = (*state->dest++ = *state->src++);

			state->check_sum += x;
			++state->counter;

			if(LIKELY(x == SOH))
				break;

			if(UNLIKELY(state->counter == 10))	// max. 9 digits + SOH
				goto MESSAGE_LENGTH_FAILURE;
		}

		// convert
		if(UNLIKELY(!convert_message_length(state)))
			goto MESSAGE_LENGTH_FAILURE;

		// store context
		parser->result.error.context.end = state->dest;

		// ensure enough space for the message body
		parser->frame.begin = state->dest = make_space(parser, state->dest, state->counter + sizeof("10=123|") - 1);

		if(UNLIKELY(!state->dest))
			return false;	// out of memory
		
		// Continue to next state
		#if defined(__GNUC__) || defined(__clang__)
		goto MESSAGE_BODY_STATE;
		#else
		// fall through
	#endif

	#if defined(__GNUC__) || defined(__clang__)
	MESSAGE_BODY_STATE:
	#else
	case 3: // message body
	#endif
		// copy
		if(state->src == state->end || !copy_chunk_cs(state)) {
			state->label = 3;
			return false;
		}

		// validate
		if(UNLIKELY(*(state->dest - 1) != SOH))
		{
			set_error(&parser->result.error, FE_INVALID_MESSAGE_LENGTH, 9);	// preserving the error context
			return false;
		}

		// update context
		parser->frame.end = parser->result.error.context.begin = state->dest;

		// prepare for trailer
		state->counter = sizeof("10=123|") - 1;
		
		// Continue to next state
		#if defined(__GNUC__) || defined(__clang__)
		goto TRAILER_STATE;
		#else
		// fall through
	#endif

	#if defined(__GNUC__) || defined(__clang__)
	TRAILER_STATE:
	#else
	case 4: // trailer
	#endif
		// copy
		if(state->src == state->end || !copy_chunk(state)) {
			state->label = 4;
			return false;
		}

		// complete message body
		parser->body_length = state->dest - parser->body;

		// validate using 32-bit comparison for speed
		if(UNLIKELY(*(const unsigned*)(state->dest - 8) != (SOH | ('1' << 8) | ('0' << 16) | ('=' << 24)) || state->dest[-1] != SOH))
			goto TRAILER_FAILURE;

		// compare checksum
		if(UNLIKELY(!valid_checksum(state)))
		{	// invalid checksum - a recoverable error
			set_error(&parser->result.error, FE_INVALID_VALUE, 10);
			parser->result.error.context.end = state->dest - 1;
		}
		else // all fine
			set_error_ctx(&parser->result.error, FE_OK, 0, EMPTY_STR);

		// all done
		state->label = 0;
		return true;

	#if !defined(__GNUC__) && !defined(__clang__)
	default:
		set_fatal_error(parser, FE_INVALID_PARSER_STATE);
		return false;
	}
	#endif

BEGIN_STRING_FAILURE:
	parser->result.error.code = FE_INVALID_BEGIN_STRING;
	parser->result.error.tag = 8;
	goto EXIT;

MESSAGE_LENGTH_FAILURE:
	parser->result.error.code = FE_INVALID_MESSAGE_LENGTH;
	parser->result.error.tag = 9;
	goto EXIT;

TRAILER_FAILURE:
	parser->result.error.code = FE_INVALID_TRAILER;
	parser->result.error.tag = 10;
	goto EXIT;

EXIT:
	parser->result.error.context.end = state->dest;
	parser->body_length = state->dest - parser->body;
	return false;
}