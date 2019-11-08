#pragma once

#include <array>
#include <functional>
#include <algorithm>

//The Nelder–Mead method of minimazing function
//https://pl.wikipedia.org/wiki/Metoda_Neldera-Meada
template <class T, int N>
void nelder_mead_minimize(std::function<T(const std::array<T, N>)>& func, std::array<T, N>& x, T initial_var) {
    std::array<std::pair<std::array<T, N>, T>, N + 1> xfn;
    
    xfn[0] = std::make_pair(x, func(x));
    for (int i = 1; i < xn.size(); ++i) {
        const auto& [xn, fval] = xfn[i];
        xn = x;
        xn[i - 1] += initial_var;
        fval = func(xn);
    }

    std::sort(xfn.begin(), xfn.end(), [](const auto &xf1, const auto &xf2){
        return xf1.second > xf2.second; //We need ascending order
    });

    std::array<T, N> u, v, w;

    for(int i =0; i < N; ++i){
        u[i] = std::reduce(xfn.cbegin(), xfn.cend(), 0.0, [](const auto&)); 
        
    }



}