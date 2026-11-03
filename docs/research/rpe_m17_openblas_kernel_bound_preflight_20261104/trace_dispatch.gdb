set pagination off
set confirm off
set breakpoint pending on
set disable-randomization off
break cblas_zgemv64_
commands
silent
printf "TRACE CBLAS layout=%d trans=%d m=%ld n=%ld mxcsr=%x\n",$rdi,$rsi,$rdx,$rcx,$mxcsr
continue
end
break zgemv_thread_t
commands
silent
printf "TRACE zgemv_thread_t\n"
continue
end
break zgemv_t_PRESCOTT
commands
silent
printf "TRACE GEMVT m=%ld n=%ld mxcsr=%x\n",$rdi,$rsi,$mxcsr
continue
end
break zgemv_n_PRESCOTT
commands
silent
printf "TRACE GEMVN m=%ld n=%ld mxcsr=%x\n",$rdi,$rsi,$mxcsr
continue
end
break zdotc_k_PRESCOTT
commands
silent
printf "TRACE complex DOT n=%ld mxcsr=%x\n",$rdi,$mxcsr
continue
end
break ddot_k_PRESCOTT
commands
silent
printf "TRACE norm real DOT n=%ld incx=%ld incy=%ld mxcsr=%x\n",$rdi,$rdx,$r8,$mxcsr
continue
end
break CDOUBLE_multiply_FMA3__AVX2
commands
silent
printf "TRACE CDOUBLE_multiply_FMA3__AVX2 n=%ld mxcsr=%x\n",*(long*)$rsi,$mxcsr
continue
end
break CDOUBLE_subtract_FMA3__AVX2
commands
silent
printf "TRACE CDOUBLE_subtract_FMA3__AVX2 n=%ld mxcsr=%x\n",*(long*)$rsi,$mxcsr
continue
end
break _Py_c_abs
commands
silent
printf "TRACE _Py_c_abs\n"
continue
end
break hypot
commands
silent
printf "TRACE hypot mxcsr=%x\n",$mxcsr
continue
end
run
