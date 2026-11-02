#!/usr/bin/env python3
"""Draw only the twenty saved comparison rows; no scientific evaluation."""
from fractions import Fraction
from pathlib import Path
import os

from trotterlib.isolated_qiskit_execution import read, save, sha


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/"artifacts/non_df_h2_process_recovery/2026-10-23"
    out=base/"figure"
    out.mkdir()
    # Matplotlib configuration/cache belongs only to this derivative output.
    os.environ["MPLCONFIGDIR"]=str(out/"mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=read(base/"comparison.json")
    styles=[("pauli_jw_PF","Pauli PF (all 14 terms)","#275DAD","o"),
            ("pauli_jw_PR","Pauli PR (7 / 14 terms)","#D46A1C","s"),
            ("df_native_PF","DF PF (all 3 fragments)","#23846A","^"),
            ("df_native_PR","DF PR (1 / 3 fragments)","#9054A8","D")]
    fig,ax=plt.subplots(figsize=(8.6,5.7),layout="constrained")
    points=[]
    for name,label,color,marker in styles:
        rows=[r for r in data["table"] if r["case"]==name and r["eligible"]]
        x=[r["epsilon_sig"] for r in rows]
        y=[float(Fraction(r["G"]["mean_exact"])) for r in rows]
        errors=[r["G"]["nominal95_halfwidth"] for r in rows]
        ax.errorbar(x,y,yerr=errors,marker=marker,label=label,color=color,lw=1.6,capsize=3,ms=6)
        points.extend(dict(case=name,epsilon_sig=r["epsilon_sig"],G_exact=r["G"]["mean_exact"],
                           G_plot=g,nominal95_halfwidth=e) for r,g,e in zip(rows,y,errors))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks([.05,.01,.005,.001,.0005],labels=["0.05","0.01","0.005","0.001","0.0005"])
    ax.set_xlabel("Complex-signal precision epsilon_sig (stricter to the right)")
    ax.set_ylabel("Total compiled RZ cost G")
    ax.set_title("H2 / 1.00 Angstrom / STO-3G: four fixed constructions\nT=0.8, q=4, delta=0.2, S2; PR r=2, K=4; P=0",fontsize=12)
    ax.grid(True,which="major",alpha=.25)
    ax.legend(fontsize=9,loc="upper left")
    fig.text(.02,-.035,"Conditional Hoeffding N with empirical numerical allowances. PR: 64 paired MC sample means;\nnominal 95% error bars do not guarantee population means or unobserved-order costs. Fixed H2 only.",fontsize=8)
    for ext in ["png","svg","pdf"]:
        fig.savefig(out/f"H2_four_fixed_G.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    save(out/"provenance.json",dict(comparison_sha256=sha(base/"comparison.json"),
        plot_source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,
        saved_values_only=True,new_scientific_computation=False,points=points,
        images={p.name:sha(p) for p in out.glob("H2_four_fixed_G.*")}))
    (out/"source.py").write_bytes(Path(__file__).read_bytes())


if __name__=="__main__":
    main()
