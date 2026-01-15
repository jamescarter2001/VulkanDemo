//
// Created by james on 10/01/2026.
//

#include "log.h"

#include <iostream>

void log::print(const char* const str) {
    std::cout << str;
}

void log::println(const std::string& str) {
    println(str.c_str());
}

void log::println(const char* const str) {
    print(str);
    std::cout << std::endl;
}
