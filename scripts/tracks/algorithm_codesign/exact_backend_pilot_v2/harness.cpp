// Synthetic-only SoPlex 7.0.0 callable-library pilot. No production interfaces.
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "soplex.h"
using namespace soplex;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
Rational readq(std::istream& in) {
    std::string s; if (!(in >> s)) throw std::runtime_error("missing rational");
    Rational r;
    if (mpq_set_str(r.backend().data(), s.c_str(), 10) != 0 || mpz_sgn(mpq_denref(r.backend().data())) == 0)
        throw std::runtime_error("invalid rational");
    mpq_canonicalize(r.backend().data());
    return r;
}
void qout(const Rational& q) { std::cout << '"' << q << '"'; }
template<class V> void vectorout(const V& v, int n) {
    std::cout << '['; for(int j=0;j<n;++j) { if(j)std::cout<<',';qout(v[j]); }std::cout<<']';
}
int main(int argc, char** argv) {
    try {
        if(argc!=2 && argc!=3) throw std::runtime_error("one input path and optional --echo-only required");
        bool echo_only=argc==3 && std::string(argv[2])=="--echo-only";
        if(argc==3 && !echo_only) throw std::runtime_error("invalid mode");
        std::ifstream in(argv[1]); int n, ma, mh;in>>n>>ma>>mh;
        if(n<1 || n>100 || ma<0 || mh<0 || ma+mh>200) throw std::runtime_error("dimensions");
        Rational c0=readq(in); std::vector<Rational> c(n), upper(n);
        for(auto& v:c)v=readq(in);for(auto& v:upper)v=readq(in);
        SoPlex s;
        bool config=s.setIntParam(SoPlex::READMODE, SoPlex::READMODE_RATIONAL)
            &&s.setIntParam(SoPlex::SOLVEMODE, SoPlex::SOLVEMODE_RATIONAL)
            &&s.setIntParam(SoPlex::CHECKMODE, SoPlex::CHECKMODE_RATIONAL)
            &&s.setIntParam(SoPlex::SYNCMODE, SoPlex::SYNCMODE_AUTO)
            &&s.setIntParam(SoPlex::OBJSENSE, SoPlex::OBJSENSE_MINIMIZE)
            &&s.setIntParam(SoPlex::SIMPLIFIER, SoPlex::SIMPLIFIER_OFF)
            &&s.setIntParam(SoPlex::VERBOSITY, SoPlex::VERBOSITY_ERROR)
            &&s.setRealParam(SoPlex::FEASTOL, 0.0)&&s.setRealParam(SoPlex::OPTTOL,0.0);
        if(!config) throw std::runtime_error("exact config rejected");
        DSVectorRational empty(0);
        for(int j=0;j<n;++j)s.addColRational(LPColRational(c[j],empty,upper[j],0));
        for(int i=0;i<ma+mh;++i) {
            DSVectorRational row(n);
            for(int j=0;j<n;++j){auto v=readq(in);if(v!=0)row.add(j,v);}
            auto rhs=readq(in);
            s.addRowRational(LPRowRational(i<ma?Rational(-infinity):rhs,row,rhs));
        }
        std::string extra;if(in>>extra)throw std::runtime_error("extra input token");
        // Echo from the solver's rational LP, not the Python input cache.
        std::cout<<"{\"echo\":{\"c0\":";qout(c0);
        std::cout<<",\"c\":[";for(int j=0;j<n;++j){if(j)std::cout<<',';qout(s.objRational(j));}
        std::cout<<"],\"U\":[";for(int j=0;j<n;++j){if(j)std::cout<<',';qout(s.upperRational(j));}
        std::cout<<"],\"lower\":[";for(int j=0;j<n;++j){if(j)std::cout<<',';qout(s.lowerRational(j));}
        for(int k=0;k<2;++k){int start=k==0?0:ma, count=k==0?ma:mh;
            std::cout<<(k==0?"],\"A\":[":",\"H\":[");
            for(int i=0;i<count;++i){if(i)std::cout<<',';vectorout(s.rowVectorRational(start+i),n);}
            std::cout<<(k==0?"],\"b\":[":"],\"f\":[");
            for(int i=0;i<count;++i){if(i)std::cout<<',';qout(s.rhsRational(start+i));}
            std::cout<<']';
        }
        std::cout<<",\"equal_lhs\":[";for(int i=0;i<mh;++i){if(i)std::cout<<',';qout(s.lhsRational(ma+i));}
        std::cout<<"]},\"exact_config_accepted\":true";
        if(echo_only){std::cout<<",\"status\":\"ECHO_ONLY\"}\n";return 0;}
        auto ts=Clock::now();auto status=s.optimize();double solve=elapsed(ts);
        std::cout<<",\"status_code\":"<<int(status)<<",\"status\":\""
                 <<(status==SPxSolver::OPTIMAL?"OPTIMAL":status==SPxSolver::INFEASIBLE?"INFEASIBLE":"OTHER")<<'"';
        double primal_time=0, dual_time=0, farkas_time=0;
        if(status==SPxSolver::OPTIMAL){
            DVectorRational p(n),d(ma+mh),r(n);
            ts=Clock::now();bool okp=s.getPrimalRational(p);primal_time=elapsed(ts);
            ts=Clock::now();bool okd=s.getDualRational(d), okr=s.getRedCostRational(r);dual_time=elapsed(ts);
            std::cout<<",\"primal_available\":"<<(okp?"true":"false")<<",\"dual_available\":"<<(okd?"true":"false")
                     <<",\"reduced_cost_available\":"<<(okr?"true":"false");
            if(okp){std::cout<<",\"primal\":";vectorout(p,n);}
            if(okd){std::cout<<",\"raw_row_dual\":";vectorout(d,ma+mh);}
            if(okr){std::cout<<",\"raw_reduced_cost\":";vectorout(r,n);}
            std::cout<<",\"backend_objective_without_offset\":";qout(s.objValueRational());
            std::cout<<",\"objective_with_exact_external_offset\":";qout(s.objValueRational()+c0);
        } else if(status==SPxSolver::INFEASIBLE){
            DVectorRational ray(ma+mh);ts=Clock::now();bool ok=s.getDualFarkasRational(ray);farkas_time=elapsed(ts);
            std::cout<<",\"farkas_available\":"<<(ok?"true":"false");
            if(ok){std::cout<<",\"raw_row_farkas\":";vectorout(ray,ma+mh);}
        }
        std::cout<<",\"timing\":{\"solve_seconds\":"<<solve<<",\"primal_acquisition_seconds\":"<<primal_time
                 <<",\"dual_acquisition_seconds\":"<<dual_time<<",\"farkas_acquisition_seconds\":"<<farkas_time<<"}}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
