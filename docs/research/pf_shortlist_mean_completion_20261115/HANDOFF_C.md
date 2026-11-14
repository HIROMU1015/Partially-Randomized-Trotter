# A → C/D：価格shortlistと同じ候補の全signal取得完了

[REPORT](REPORT.md)／[6packet SOURCE・ID一覧](PACKET_INDEX.json)／[78round表](ROUND_SUMMARY.csv)。新科学SOURCE fb9a1878b677b7ee116b52e804e5a01cdaa96004、packet carrier06a15ee45beb177c018f6260ea4a5ac8cf66cfe6。Full PF/B0I p7×S2 16h/7・Y4 16h/3・M5 16hの6候補が各13round完成。48新規PASS、30保存再利用、B intake6/6 PASS。

C855573064のCELL_PLAN_V2、C0b4b2d94の14構成対応表、A staging217eab647と同configuration/physical IDを保つ。価格m4/m5/m6のcandidateは同じであり、S2・幅.1の全13roundとshortlistを方式名だけで混ぜる必要はなくなった。

m0共有2点はphysical ID一致を前提にユーザーが再利用を明示指定した保存roundで、別schedule全体の流用ではない。各actual幅・q・signed weights・Full/B0I scalarを元configurationから取得する。表示shortlist幅を全roundへ機械的に適用しない。

元configuration先頭は Full:11cf8ca6/a799697f/70f7b317、B0I:a6e55777/defc5e88/424eecbf。Bが使うtotal-view IDとの対応もPACKET_INDEXにある。Gで結合するときは元physical ID/round/X/Y/law/profileを照合し、total-viewの新IDだけで別候補と判定したり、元configurationを上書きしたりしない。

今回価格・fit/holdout・E[C]/G・rankingは0。全signalの局所数値gate通過は必要Nを保証せず、C/Dの価格資格を自動付与しない。次は別承認のB MSE/Nと、C/Dの同候補・全roundの期待compiled価格証拠を揃え、採択policyの下で会計する。

ロードマップ統合担当へ：信号欠測48・staging未完成というブロッカーは解消。固定REPORTを参照し、「Full PF全13round照合」「Truncatedの信号/価格shortlist不一致」の状態と次依存を更新できる。Aはロードマップ本体やC/Dのbranchを変更していない。

ユーザーが次の科学scopeを判断・承認し、B/C/Dが各担当範囲で進める。Aはcommit/push・remote照合後mandatory STOP。
