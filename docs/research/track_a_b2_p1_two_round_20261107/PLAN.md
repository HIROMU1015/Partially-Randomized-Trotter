# B2 P1・m0/m17限定実行

ユーザーが2026-11-07にP1採用と本限定科学実行を明示承認した。
条件の正本は`7bc952772e451f27c1654dab6803192f2b204c9f`のP1 REPORTとユーザー指示。
保存private state/provenance/HDF5は原本不変更・非公開。旧DF12の固定Git入力を再利用する。

1. 最小adapterと変更部分の人工テストを検証し、SOURCEをcommit固定する。
2. private入力・sector/order・raw H_phys/DF12 hashを照合。不一致ならSTOP。
3. 代表primitive/1 outer stepの未compile controlled correctnessを検証する。
4. B2 LD3/r4/K2、delta=.02、m=0と17だけの5信号を既存平均kernelと独立momentsで評価する。
5. u=1e-8、独立照合u/10、P1の6誤差cap・reserve・均等alphaを変更しない。
6. 不適格時は停止理由、完了時は2点の条件付き適格性を保存・公開してmandatory STOP。

shot、compile/価格、sampling、新state solve/SCF/DF、他round、探索、branch成功判定、詳細科学レビューは行わない。
nativeの長round入力は既存canonical平均lawに基づく理想programのQ乗という条件付き数値診断。
代表primitive PASSを全supportの浮動小数点誤差証明や131072-step実回路保証へ昇格しない。
