#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include "leibnizPI.h"

double compute_pi_baseline(size_t N)
{
    double pi = 0.0;
    double  sign=1;
    for(int i=0; i<N; i++) {
        pi+=(double)(sign/(1+2*i));
        sign*=-1;
    }
    return 4*pi;
}

double compute_pi_openmp(size_t N,int threads)
{
    double pi = 0.0;
    double  sign=1;
#pragma omg parallel for num_threads(threads) reduction(pi)
    for(int i=0; i<N; i++) {
        sign = i%2==0 ? 1:-1;
        pi+=(double)(sign/(1+2*i));
    }
    return 4*pi;
}

double compute_pi_avx(size_t N)
{
    double pi=0.0;
    __m256d sign1,one,ans,data;
    sign1=_mm256_set_pd(-1.0,1.0,-1.0,1.0);
    ans=_mm256_set_pd(0,0,0,0);
    one=_mm256_set_pd(1,1,1,1);
    for(int i=0; i<=N-4; i+=4) {
        data=_mm256_set_pd(7+2*i,5+2*i,3+2*i,1+2*i);
        data=_mm256_div_pd(one,data);
        data=_mm256_mul_pd(data,sign1);
        ans=_mm256_add_pd(ans,data);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp,ans);
    pi+=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    return 4.0 * pi;
}

double compute_pi_avx_unroll(size_t N)
{
    double pi=0.0;
    __m256d sign1,one,ans,ans2,ans3,ans4,data,data2,data3,data4;
    sign1=_mm256_set_pd(-1.0,1.0,-1.0,1.0);
    ans=_mm256_set_pd(0,0,0,0);
    ans2=_mm256_set_pd(0,0,0,0);
    ans3=_mm256_set_pd(0,0,0,0);
    ans4=_mm256_set_pd(0,0,0,0);
    one=_mm256_set_pd(1,1,1,1);
    for(int i=0; i<=N-16; i+=16) {
        data=_mm256_set_pd(7+2*i,5+2*i,3+2*i,1+2*i);
        data=_mm256_div_pd(one,data);
        data=_mm256_mul_pd(data,sign1);
        ans=_mm256_add_pd(ans,data);

        data2=_mm256_set_pd(15+2*i,13+2*i,11+2*i,9+2*i);
        data2=_mm256_div_pd(one,data2);
        data2=_mm256_mul_pd(data2,sign1);
        ans2=_mm256_add_pd(ans2,data2);

        data3=_mm256_set_pd(23+2*i,21+2*i,19+2*i,17+2*i);
        data3=_mm256_div_pd(one,data3);
        data3=_mm256_mul_pd(data3,sign1);
        ans3=_mm256_add_pd(ans3,data3);

        data4=_mm256_set_pd(31+2*i,29+2*i,27+2*i,25+2*i);
        data4=_mm256_div_pd(one,data4);
        data4=_mm256_mul_pd(data4,sign1);
        ans4=_mm256_add_pd(ans4,data4);
    }
    double tmp[4],tmp2[4],tmp3[4],tmp4[4];
    _mm256_storeu_pd(tmp,ans);
    _mm256_storeu_pd(tmp2,ans2);
    _mm256_storeu_pd(tmp3,ans3);
    _mm256_storeu_pd(tmp4,ans4);

    pi+=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    pi+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];
    pi+=tmp3[0]+tmp3[1]+tmp3[2]+tmp3[3];
    pi+=tmp4[0]+tmp4[1]+tmp4[2]+tmp4[3];

    return 4*pi;
}
