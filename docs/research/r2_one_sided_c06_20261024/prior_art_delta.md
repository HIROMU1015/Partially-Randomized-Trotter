# 既存G6・R3・先行研究との限定差分照合

C0/C0.5の文献索引を継承し、下表の関係節だけ一次本文で再確認した。文献全体の再調査、詳細科学レビュー、新規性の正式採択は行っていない。表にないGreen/Euler/compact BCH等はC0.5の固定帰属を維持する。

|原典・今回の読取位置|候補への対応|R2-CSとして残る独立差分|
|---|---|---|
|[Cugini–Atif–Subaşı 2603.13495v1](https://arxiv.org/html/2603.13495v1), II Thm.1 Eqs.9–12、IV.1/IV.2はC0.5確認範囲を継承|費用と二次momentのCauchy–Schwarz下界。G6固定辞書版は既存特殊化|P-Aの下界適用自体に新規性なし。全native価格の取得を無料にする根拠にもならない|
|[Casaresら SPRINT 2606.30741v1](https://arxiv.org/html/2606.30741v1#S4), IV、特にStep7|誤差削減とremainder実装費用を合わせて戦略を選ぶ|誤差・normalization・費用を結合する目的だけでは差分にならない。新しい安価なcertificateが必要|
|[Poianiら MF-BAI 2406.03033v1](https://arxiv.org/html/2406.03033v1#S2), Introductionのcost/maximal-bias仮定と§2|fidelity費用とbias保証を与えた選択問題|未知bias/取得費用を既知入力に移すだけでは新機構にならない。今回その仮定を安価に構成する新結果はない|
|R3固定STOP／C0.5_PRIOR|quantum固有情報、安全棄却、abstain、有料照会、regret、少数評価|P1やfeature名の変更はSTOPを解消しない|
|G6_NOTE・W2B・W2Cの既存構成|local query/P5、認証N、logical thinning、固定辞書下界、区間比較、全trial費用条件|P-A/P-Bの閉じる部分は既存成果。第2研究へ二重計上しない|
|PR2/RTE SOURCEとC0.5|同じmicrostep法の共有、signed schedule、整数配分|P-Cはcacheによる工学的取得共有。新しい適格性・順位証明ではない|

重要な区別は、数式が書けるか、入力を取得できるか、取得が安価か、選択が可能になるか、新しい方法差か、である。前三者の一部が成立しても最後二つは自動ではない。

今回の限定範囲では、既存法へ既知law特徴を入力する以上の独立選択機構は特定できない。これは文献上の全手法を網羅した同値性定理ではなく、C0.6の終了推奨の根拠である。今後の同種調査や数値pilotを推奨しない。
