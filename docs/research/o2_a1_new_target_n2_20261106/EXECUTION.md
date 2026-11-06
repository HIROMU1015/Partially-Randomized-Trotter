# A1新finite target：N2限定実行仕様

2026-11-06、SERVER1 GPUA100-gleap-01。ユーザーの最新指示でN2のみを許可した。
N1結果744e1a09661ebba6841fe5faf456f88c41c47780、科学SOURCE b6dbec9f80da35a88fba28795ad02ca74c9adfa0、契約2b2e6c1bc3b2d201d1a6d1238172187d68aea6abを参照する。
正式共通指示2026-10-25と実際のmainのAGENTS.mdを読み、既存のSTOPを履歴として保全する。N2の許可はN3やε/α/測定予算の採択ではない。

対象：保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子number sector（70次元）、L_D3/m0/q1/δ.02/r1/K2/coefficient_atol0、X/Y。
N1のH・partition・preparation・tail・actual-Q・policy identityと全payloadを固定inputにする。N1 split/preparation/CDF/policyを再生成しない。

## 既存stateと再利用資産

[STATE_INVENTORY](STATE_INVENTORY.json)はmainおよび元WP11の既存DF ground cache129件のhash/identity/headerを読み取り専用で記録する。vectorやenergyはこの調査で評価していない。
全4電子sector hashに対応するcacheは0件で、N1 artifactにもvectorはない。既存の36次元spin sector等を同一sectorと読み替えない。
実行時に同じ全Hのground-cache hashとsector hashで再照合し、適格vectorがなければ既存`solve_df_ground_state`を一回だけ使う。旧opt2値の無条件流用はない。

再利用：既存NPZ loader、DFHamiltonian、PhysicalSector.number_sector、solve_df_ground_state、fermionic excitation table、OpenFermion→Qiskit bit reversal、Hadamard wrapper/phase helper/RTE event builder、shared_cpu、N1 profile、既存directed mpmath interval helper。
新module/runnerはSOURCE固定・N1 payloadのread-only検査と、正確なstate証拠の不足を補う小adapterである。共有coreは変更しない。

## State検証と数値保証の範囲

全保存DF Hをそのままsolverへ渡す。H_DやPF有効状態を入力しない。適格cacheがなければPython matvec backend・既存tol1e-10・deterministic all-ones v0で一回solveする。
70次元ではJIT/parallel dispatchより直接matvecが適するためのbackend選択で、CPU/RAM/threadの人工上限は設定しない。
独立した全70次元spectral検査としてNumPy eighを一回使う。これはsolverのretryではなく、最低固有空間・第一励起を全sectorで確認するための処理で、科学callに別記する。

保存係数のbinary64をexact rationalとして、既存excitation tableを使い
H=cI+one_body+Σλ_l A_l²を厳密に組み立てる。追加1/2、係数threshold、非Hermitian/complex入力への自動symmetrizeは行わず、不適合ならSTOPする。
この実装は今回のreal Hermitian inputに限定する。
浮動小数点固有値d・固有vector Qを証拠候補とし、Gram defect QᵀQ−IとH−Q diag(d)Qᵀの全演算をFractionで検査する。
対称行列の最大絶対row-sumはoperator norm上界なので、それぞれg,eとし、g<1を要求する。
polar factor Oについて ||Q−O||≤g/(1+sqrt(1−g))=η、||Q||≤sqrt(1+g) より、
||H−O diag(d)Oᵀ||≤e+max|d| η(sqrt(1+g)+1)=ρ。
既存interval helper（mpmath1.3.0 iv100dps・外向き256bit endpoint）でρを上側に評価し、ordered eigenvalue区間[d_i−ρ,d_i+ρ]を得る。
BLAS残差だけの条件付きenclosureをそのまま採用せず、exact assembly/Gram/reconstructionへ適応した理由はここにある。

l1>u0で一意groundの分離を確認し、保存vectorをexact normで正規化したRayleigh値rから
excited weight≤(r−l0)/(l1−l0)、任意の同じideal unitaryに対するX/Y mean誤差≤2sqrt(weight)を求める。
これは採択草案§6の受理経路の具体化であり、residualだけのstate誤差保証ではない。必要な分離が得られなければUNKNOWN/STOP。精度目標・予算は自動採択しない。
完全vector/E/gap/eigensystemは`engine_internal/`とstate auditだけに置き、estimator/feature/fit/配分へ渡す経路を作らない。

## Engineの境界

人工の1-system-qubit回路で既存Hadamard wrapper本体、controlled scalar/identity phase、RTE n=2の−1と負符号、X/Y、Sdg→H、bit0→+1/bit1→−1を確認する。
独立なFermionOperatorによる小模型の照合と全basis indexのbit reversalを検査する。人工テストは実分子のpath検査ではない。
実行時はN1の保存law/policy identityとstateのbit reversalをbindし、full interrogationをbuildせず対応範囲を記録する。

独立uniform M=2^53 wordとthreshold U<p_hatなら、Bernoulli meanの座標grid誤差≤2/Mという条件付き項を記録する。RNG/bit生成自体は実行しない。
既存fixed_signal_evaluationやSpectrumをそのままA1 engineとして使うと、再preparationまたはalgebraic referenceの作用を使い、保存selected full列への直接の対応を確認できない。
そのため旧bias/許容値を借りず、selected gate applicationとBorn probabilityのuniform数値誤差、normalization/range guardの実装bindingが未閉鎖なら、**total b_engine=UNKNOWN、N3_ready=false**とする。
N2では実分子full回路を使わず、この制約を成功扱いで隠さない。後段のengine実装検証scopeを別途定める必要がある。
草案のb_engine≤ε_axis/8、ε候補1/4（閾値候補1/32）は参照のみ。新しいε/α、train/pilot/cal/prod数やcontrolは採択しない。

## 固定・実行・STOP

人工テスト後に新SOURCEと全依存環境をcommit・公開照合し、cleanな専用worktreeから一回だけ起動する。
共有venv/package/cacheは変更せず、自分専用`/tmp/a1-n2-cache-20261106/`のみ使用する。

```bash
MPLCONFIGDIR=/tmp/a1-n2-cache-20261106/mpl \
NUMBA_CACHE_DIR=/tmp/a1-n2-cache-20261106/numba \
XDG_CACHE_HOME=/tmp/a1-n2-cache-20261106/xdg \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B \
scripts/tracks/algorithm_codesign/run_a1_new_target_n2.py \
 --profile-lock docs/research/o2_a1_new_target_n2_20261106/PROFILE_LOCK.json \
 --output-dir artifacts/o2_a1_new_target/2026-11-06/n2
```

[新module](../../../src/trottertracks/algorithm_codesign/a1_new_target_n2.py)、[runner](../../../scripts/tracks/algorithm_codesign/run_a1_new_target_n2.py)、[人工test](../../../tests/tracks/algorithm_codesign/test_a1_new_target_n2.py)。
SOURCE/STATE_RESULT/ENGINE_RESULT、完全engine用vector・spectral witness、CALL_COUNTS/RUN.log/STOPをfresh directoryに保存する。
不整合で条件変更・retry・spin-sector fallbackをしない。N1再計算、RNG、Hadamard shot、実分子full build/compile、variance/N/費用評価はguardで拒否する。
結果をcommit/pushし、remote本文/blob/treeを照合してmandatory STOP。N3、18round展開、C1解凍や重要科学レビューへ自動進行しない。
