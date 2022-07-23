#ifndef UTILS_H
#define UTILS_H

#include <utility>

template<typename T1, typename T2>
class MyPair {
public:
	T1 first;
	T2 second;
public:
	MyPair(T1& first, T2& second) {
		this->first = first;
		this->second = second;
	}
	MyPair(T1&& first, T2&& second) {
		this->first = first;
		this->scond = second;
	}
	MyPair(T1&& first, T2& second) {
		this->first = first;
		this->scond = second;
	}
	MyPair(T1& first, T2&& second) {
		this->first = first;
		this->scond = second;
	}
	MyPair(MyPair& other) : MyPair(other.first, other.second) {}
	MyPair(MyPair&& other) : MyPair(std::move(other.first), std::move(other.second)) {}
};

#endif // !UTILS_H