#include "cpuinfo.h"

const CpuInfo::InstructionSet_Internal CpuInfo::CPU_Rep;

#if (defined(linux)&&defined(__GNUC__))
void cpuid(int info[4], int InfoType) {
	__cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

CpuInfo::InstructionSet_Internal::InstructionSet_Internal() :f_1_ECX_{ 0 }, f_1_EDX_{ 0 }, f_7_EBX_{ 0 }, f_7_ECX_{ 0 }, f_81_ECX_{ 0 }{
	int info[4] = { 0,0,0,0 };
	cpuid(info, 0);
	int nIds_ = info[0];
	if (nIds_ >= 1) {
		cpuid(info, 1);
		f_1_ECX_ = info[2];
		f_1_EDX_ = info[3];
	}
	if (nIds_ >= 7) {
		cpuid(info, 7);
		f_7_EBX_ = info[1];
		f_7_ECX_ = info[2];
	}
	cpuid(info, 0x80000000);
	int nExIds_ = info[0];
	if (nExIds_ >= 0x80000001) {
		cpuid(info, 0x80000001);
		f_81_ECX_ = info[2];
	}
}

bool CpuInfo::MMX() { return CPU_Rep.f_1_EDX_[23]; }
bool CpuInfo::SSE() { return CPU_Rep.f_1_EDX_[25]; }
bool CpuInfo::SSE2() { return CPU_Rep.f_1_EDX_[26]; }
bool CpuInfo::SSE3() { return CPU_Rep.f_1_ECX_[0]; }
bool CpuInfo::SSSE3() { return CPU_Rep.f_1_ECX_[9]; }
bool CpuInfo::SSE41() { return CPU_Rep.f_1_ECX_[19]; }
bool CpuInfo::SSE42() { return CPU_Rep.f_1_ECX_[20]; }
bool CpuInfo::AES() { return CPU_Rep.f_1_ECX_[25]; }
bool CpuInfo::AVX() { return CPU_Rep.f_1_ECX_[28]; }
bool CpuInfo::FMA() { return CPU_Rep.f_1_ECX_[12]; }
bool CpuInfo::RDRAND() { return CPU_Rep.f_1_ECX_[30]; }
bool CpuInfo::AVX2() { return CPU_Rep.f_7_EBX_[5]; }
bool CpuInfo::BMI1() { return CPU_Rep.f_7_EBX_[3]; }
bool CpuInfo::BMI2() { return CPU_Rep.f_7_EBX_[8]; }
bool CpuInfo::ADX() { return CPU_Rep.f_7_EBX_[19]; }
bool CpuInfo::SHA() { return CPU_Rep.f_7_EBX_[29]; }
bool CpuInfo::PREFETCHWT1() { return CPU_Rep.f_7_ECX_[0]; }
bool CpuInfo::AVX512F() { return CPU_Rep.f_7_EBX_[16]; }
bool CpuInfo::AVX512CD() { return CPU_Rep.f_7_EBX_[28]; }
bool CpuInfo::AVX512PF() { return CPU_Rep.f_7_EBX_[26]; }
bool CpuInfo::AVX512ER() { return CPU_Rep.f_7_EBX_[27]; }
bool CpuInfo::AVX512VL() { return CPU_Rep.f_7_EBX_[31]; }
bool CpuInfo::AVX512BW() { return CPU_Rep.f_7_EBX_[30]; }
bool CpuInfo::AVX512DQ() { return CPU_Rep.f_7_EBX_[17]; }
bool CpuInfo::AVX512IFMA() { return CPU_Rep.f_7_EBX_[21]; }
bool CpuInfo::AVX512VBMI() { return CPU_Rep.f_7_ECX_[1]; }

bool CpuInfo::FMA4() { return CPU_Rep.f_81_ECX_[16]; }
bool CpuInfo::SSE4a() { return CPU_Rep.f_81_ECX_[6]; }

CpuInfo::~CpuInfo() {}