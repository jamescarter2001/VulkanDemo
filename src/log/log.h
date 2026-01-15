//
// Created by james on 10/01/2026.
//

#pragma once
#include <string>

class log {
public:
    static void print(const char* str);
    static void println(const std::string& str);
    static void println(const char* str);
};
