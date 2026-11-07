# CT-1 scope（合成前）

基準f5f16fad64748236b0160284dc6c3017974b0132、SERVER 2、独立branch codex/pr-reference-ct1-20261107。
固定H4/1Å/STO-3G/DF12、exact ground供給/準備費除外、Partial S2/LD3/h=.02/K2、ordinary control、canonical finite-bit law、energy1.5936e-4 Ha・失敗率.05・phase.4、Qiskit1.3/compiler Aを維持。

保存pilotのq1/r1/j0、rare0/1の2 trajectoryを価格を見ずに選ぶ。既存Reference X QPYがある最小の各stratum代表だからであり、新samplingはしない。主比較B′/Reference、X/Y。B′の4-CX ZZ、Reference F/G actual inverse/vacuum P/2 register/8-CX CSWAP、reset8、D融合/basis共有/scalar/event/wrapperを変更しない。

必要な新whole native compileはB′2 trajectory×X/Y=4、Reference同2本のY=2、計6。Reference Xは保存QPYを使う。全て保存X/Y価格と照合する。local構造は保存catalogueを使い新local compileを避ける。新local回路は保存native gate列の再構成とClifford+Tへの置換だけ。

local代表はD0 Gaussian primitive、basis D0/inverse、D diagonal/scalar、RTE order0/2のZ/ZZ中央、event/scalar、vacuum P、CSWAP、H/Sdg。必要keyはこれらと有限whole8回路の依存閉包の和集合。whole依存keyは全置換に必須であり、構造が同じkeyや異なるtrajectoryを追加しない。合成前にPREDECLAREDへidentity/全key/role/一様exponent/根拠/既存・新規区分を保存する。

uniform exponentはCT-0 q1両arm両軸のworst L/固定N headroomを満たす保存仕様の最も緩いもの。全18roundの採択・最適precisionではない。4 legacy recordを除外し、合成は固定missing keyだけ。失敗後のprecision変更・key追加はしない。

strict binary RZ、word逆順、actual adjoint、Wを維持しphase fitなし。local full-space operator、exact ring/interval guard、whole状態と測定信号・telescoping・raw disk/旧N条件を区別する。17qubit dense whole matrixは作らない。シミュレーションは各wordの同一2×2演算への圧縮を使用できるが、物理wordの変更/融合/最適化ではないこととemitter対応を検査する。

finite pilotだけのT/word件数を保存。全support vacuum価格、18round総T、T-depth/FT depth/最小T/最適性は未資格のまま。旧SOURCE/raw/result/STOPを保全する。終了または本質的不整合/guard失敗/対象外拡張が必要な時点でmandatory STOP。次担当GPT／ユーザー。
