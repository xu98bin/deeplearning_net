#ifndef CPUINFO_H
#define CPUINFO_H

#include<bitset>

#if ((defined(_WIN32)||defined(_WIN64))&&(defined(_MSC_VER)))
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#define cpuid(info,x) __cpuidex(info,x,0)
#elif (defined(linux)&&defined(__GNUC__))
#include <x86intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <cpuid.h>
void cpuid(int info[4], int InfoType);
#endif

class CpuInfo {
private:
	class InstructionSet_Internal {
	public:
		InstructionSet_Internal();
	public:
		std::bitset<32> f_1_ECX_;
		std::bitset<32> f_1_EDX_;
		std::bitset<32> f_7_EBX_;
		std::bitset<32> f_7_ECX_;
		std::bitset<32> f_81_ECX_;
	};
	~CpuInfo();
	CpuInfo(const CpuInfo&) = delete;
	CpuInfo(CpuInfo&&) = delete;
	CpuInfo& operator=(const CpuInfo&) = delete;
	CpuInfo& operator=(CpuInfo&&) = delete;

	static const InstructionSet_Internal CPU_Rep;
public:
	static bool MMX();
	static bool SSE();
	static bool SSE2();
	static bool SSE3();
	static bool SSSE3();
	static bool SSE41();
	static bool SSE42();
	static bool AES();
	static bool AVX();
	static bool FMA();
	static bool RDRAND();
	static bool AVX2();
	static bool BMI1();
	static bool BMI2();
	static bool ADX();
	static bool SHA();
	static bool PREFETCHWT1();
	static bool AVX512F();
	static bool AVX512CD();
	static bool AVX512PF();
	static bool AVX512ER();
	static bool AVX512VL();
	static bool AVX512BW();
	static bool AVX512DQ();
	static bool AVX512IFMA();
	static bool AVX512VBMI();

	static bool FMA4();
	static bool SSE4a();
};

#endif