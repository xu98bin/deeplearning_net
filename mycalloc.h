#pragma once
#include <cstdlib>

class XCalloc {
public:
	static int num_calloc;
public:
	static void* xcalloc(size_t count, size_t size);
	static void xfree(void* p);
};
