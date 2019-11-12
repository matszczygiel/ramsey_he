#pragma once

#include <array>
#include <functional>
#include <algorithm>
#include <type_traits>
#include <cassert>

//The Nelder–Mead method of minimazing function
//https://pl.wikipedia.org/wiki/Metoda_Neldera-Meada
template <class T, int N>
T nelder_mead_minimize(std::function<T(const std::array<T, N>&)>& func, std::array<T, N>& x, 
T initial_var, const double& alpha, const double& gamma, const double& beta, const T& epsilon, int max_iters) {
    assert(alpha > 0.0);
    assert(gamma >= 0.0);
    assert(1.0 > beta && beta > 0.0);
    assert(max_iters > 1);
    
    std::array<std::pair<std::decay_t<decltype(x)>, T>, N + 1> xfn;
    const auto sort_xfns = [&xfn](){
    std::sort(xfn.begin(), xfn.end(), [](const auto &xf1, const auto &xf2){
        return xf1.second > xf2.second; //We need ascending order
    });};

    T current_minimum;
    
    xfn[0] = std::make_pair(x, func(x));
    for (int i = 1; i < xn.size(); ++i) {
        const auto& [xn, fval] = xfn[i];
        xn = x;
        xn[i - 1] += initial_var;
        fval = func(xn);
    }


    sort_xfns();

    std::tie(x, current_minimum) = xfn.back();

    
    std::array<T, N> u, v, w;
    int iteration = 1;
    for(; iteration <= max_iters; ++iteration){

    for(int i =0; i < N; ++i){
        u[i] = std::reduce(xfn.cbegin(), xfn.cend(), T(0.0), 
        [&i](const auto& x1, const auto& x2){return x1.first[i] + x2.first[i]}); 
        v[i] = (1.0 + alpha) * u[i] - alpha *xfn[0].first[i];
        w[i] = (1.0 + gamma) *v[i] -gamma * u[i]; 
        
    }

    const auto f_v = func(v);
    const auto f_w = func(w);

    if(f_v < xfn[N].second){
        if(f_w < xfn[N].second){
            xfn[0].first = w;
            xfn[0].second = f_w;
        }else {
            xfn[0].first = v;
            xfn[0].second = f_v;
        }
    }
    else if(f_v >= xfn[N].second && f_v <= xfn[1].second){
        xfn[0] = {v, f_v};
    }
    else  {
        if(f_v <= xfn[0].second){
            xfn[0].first = v;
            xfn[0].second = f_v;
            w = beta * xfn[0].first + (1.0 - beta) * u;
            const auto f_w_2 = func(w);
            if(f_w_2 <= xfn[0].second){
                xfn[0].first = w;
                xfn[0].second = f_w_2;
            } else {
                for(int i = 0 ; i < N; ++i){
                    for(int j = 0 ; j < N; ++j)
                    {
                        xfn[i].first[j] = 0.5 * (xfn[i].first[j] + xfn[N].first[j])
                    }
                    xfn[i].second = func(xfn[i].first);
                }
            }
        }

    }
    sort_xfns();
    const auto last_min = current_minimum;
    std::tie(x, current_minimum) = xfn.back();
    if(std::abs(last_min - current_minimum) < epsilon)
        break;

    }



    return current_minimum;
}