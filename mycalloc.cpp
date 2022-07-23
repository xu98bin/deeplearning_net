#include "mycalloc.h"

void* XCalloc::xcalloc(size_t count, size_t size) {
	num_calloc++;
	return calloc(count, size);
}

void XCalloc::xfree(void* p) {
	if (p != nullptr) {
		num_calloc--;
		free(p);
	}
	p = nullptr;
}

int XCalloc::num_calloc = 0;