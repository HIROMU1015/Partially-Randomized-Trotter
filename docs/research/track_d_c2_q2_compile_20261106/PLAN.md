# Track D C2：q=2 whole-path、4 wrapper compile固定scope

2026-11-06 JST。基点は[B1/C1結果](../track_d_b1_c1_validation_20261106/RESULT.md) `d3913866e6ade4ab97ac02645fed91b2d5fcfa75`。ユーザーの今回の明示許可に従い、同じq2固定fixtureの4 wrapperを既存P0で一回ずつcompile・postcheckする。旧SOURCE・入力・result・grant・ledger・marker・STOPを変更・再消費しない。実行前にSOURCEをcommit固定する。

## 固定fixture・入力identity

linear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8／元frame順LD2／S2/s1/r32/K0、各h=δ=.1。sample00→01の64 events、q2、総時間.2。V=U01 U00、B=U00、A=U01、cutはcompleteな最初のh-step終了後（head／逆順head／scalar・identity込み）。χ=−.4574879096539928 rad、scalar／identity各2回、一つのwhole-path Reference gadget。元H厳密ground仮定、準備費除外。

元H packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。C1科学SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8`、Stage A SOURCE／結果 `2dd012dc77dcc877d5918c7d212819eb79b97557`／`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`。元C SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`と保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f`の既存closure・package pinsを再利用する。

C1の`C1_payload.tar.gz`はSHA256 `2cef1b138af95b75f50d80ee89fdc7bb7aee13d133fa05b38d91a82f80ec01d9`。C1 metadata／archiveを固定commitとbyte照合し、member11件（raw QPY10・references.npz）をfresh outputへ同一bytesで復元する。新しいraw生成、head再構成、cut最適化、basis／fusion変更をしない。

## precompile → 4 compile／postcheck

1. 既存B1/C1 preflightで元SOURCE143・入力16・保存96 binding・固定10 pins等を再照合する。B1科学評価やC1 state-actionは再実行しない。
2. C1の22 check＋paired X/Y PASSを同一raw／reference bytesで引き継ぐ。追加のsmall-operator検査はAB=V、actual A†／inverse、vacuum ray／χ、cached A/B/adjoint、強いsymmetric coreの両branch=diag(I,V)。scalar/identity2回・χ合成metadataを維持する。
3. PASSした場合だけsymmetric-directional X/Y、Reference X/Yの4 full-wrapperを既存`compile_one`で各一回compile。P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topology-free。既存`compile_checked`→MPFR160 declared-phase→位相保持unroll。旧関数を変更しない。
4. 既存`verify_one`／`check_compiled`でrawとの位相付き対応と保存CAR由来X/Y observableを照合する。directは全512 operator、Referenceはreference/ancilla=0で全256 system入力列・各131072出力振幅、saved raw isometry／actual-adjoint branchを使う。17-qubit全Operatorは作らない。
5. compiled方式間paired X/Yと、vacuum入力に対するX+iY=e^{iχ}を確認する。complex128、component／unitarity abs1e−10、signal／数値Frobenius2e−10、rtol0、phase fitなし。失敗後に緩和しない。A†単体の追加compileは行わず、A†を含むfull-wrapper作用を照合する。

## 公平性・q1との比較

強いsymmetricはC1の保存core連結のまま。各stepの既存Gaussian非制御・directional center・relative fusion・tail basis共有・global phaseを維持する。両方式へ同一P0を適用し、compilerが可能な最適化を許す。固定rawの比較であり、新しいwhole-path baseline構成やcross-step手動最適化は追加しない。

4回路のRZ／CX count、RZ／CX depth、total depth、qubits、sizeを記録し、各axisでReference−symmetricを出す。q1はStage Aの保存sample00／01 X/Yの個別価格だけを引用する。q1は各h-stepのevent16 cut、q2はstep境界cutであり、q・長さ・分割が同時に異なる。q1価格の和や単純q倍をq2価格とはしない。母平均、全law、長q、N、総RPE費用、主資源指標の採択へ拡張しない。

## 実行・資源・STOP

fresh outputは`artifacts/track_d_c2_q2_compile/2026-11-06/run01`一回。実資源は既存`cpu_observation`でaffinity／idle CPU／空きRAMを読み、独立task数4と空き資源からprocess数、Reference2件のthreads配分を選ぶ。small matrix／compilerは既存BLAS1方針、Referenceのstate-action／Gramを並列化する。CPU/RAM/worker/wallの人工capは設けず、共有設定・共有venv・他job・他ユーザーへ干渉しない。AI sub-agentは使わない。

SOURCE／入力／位相／correctness不一致、公平なbaseline構築不能は新callを停止し、開始済み処理のpartialを保全する。無断retry、wrapper置換、条件・閾値緩和は0。SOURCE、原raw／compiled QPY、observable、compile／check記録、比較、inventory、result／STOPを保存する。実行前後にSOURCE／入力と復元raw bytesを再照合する。

成功・不成立とも専用branch `codex/track-d-c2-q2-compile-20261106`へcommit/pushし、remote SHAと公開成果物を照合後mandatory STOP。B2、新sampling、別path／q、長q compile、N、全RPE費用、追加科学検証・詳細レビューへ進まない。次担当はユーザー／GPT。今回の科学実行許可は新grant・承認証明の発行を要しない。
