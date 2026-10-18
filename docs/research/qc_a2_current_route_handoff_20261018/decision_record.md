# A2 Phase B後：現行実用化経路への追加投資見送り

2026-10-18 JST。**ユーザーが承認した方針の記録・保全のみ**。[machine決定](decision.json)、[成果・限界](achievements_and_limits.md)、[終了・再開ハンドオフ](handoff.md)、[証拠索引](evidence_index.json)、[保全・静的照合](preservation_and_consistency.json)、[公開照合](publication_receipt.json)。新しい科学レビュー・証明・数値解析ではない。

## 採択と対象

ユーザーの「A2 Phase Bレビュー結果の方針固定・研究終了ハンドオフ」により、**現行A2の実用化・十分step改善経路は今回のpilotで区切り、追加探索・投資を見送る**。対象は係数三角capによるPSD化、raw K0行和とCauchy型有限剰余を用いるRF0/RF1/EW2のstep優位を目指した現在の経路である。

**A2全体の正式採択・全面終了とは区別する。** A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行を保持し、RF定理の数学的成立を撤回しない。A2一般や低energy保証一般のNO-GOを認定しない。①PRリソース評価・②PR内部改善・③M1-S、④内の他の既存停止・保留状態にも介入しない。

## 根拠の来歴

[GPT重要科学レビュー原本](inputs/qc_a2_phase_b_results_scientific_review_20261018.md) をローカルから取得・全文読取し、bytesを変更せず保存した。原本の「推奨／採択前／STOP」という記述はレビュー時点の履歴として残す。その推奨を今回ユーザーが正式に採択したことをこの新規記録で示す。Codexがレビューを再実施・追認するための新しい証明を行ったものではない。

Phase Bの固定H4 DF／1.00 Å／STO-3G／保存rank12、total N=4・保存8 modes・全12 blocks、one_body→block0〜11のpositive S2、trial0〜3、d=0.001Ha・α_spec=0.1の比較では、RF1/EW2の十分stepが約1.71e-4 Ha^-1、centered構造付きCbar2対照が約1.69e-2 Ha^-1だった。約99倍という概数は完了レビューの引用であり、ここで比を再計算していない。独立checker受理、全route ENCLOSED、比較 `COUNTER_LARGER_SUFFICIENT_STEP` を保持する。

採択根拠には、現認証値・既存保証式を固定したd/α変更での救済が支持されないというレビュー§5の紙上診断、同Λ・同適用域のままtrialやK_SOSだけを改善しても今回の対照stepに届かないという§6の限定判断を含む。現在のh_Rを直接止めたのが適用域端だったとは記録しない。これらは原レビューの判断として参照し、新たなSOURCE認証済み定理・数値実験へ格上げしない。

約343秒のpipeline時間の大部分は構造付き交換子対照の生成・検証であり、全時間を相対形式固有費用へ帰属させない。step優位の負の結果を保存しつつ、古典取得費込みで全指標に一様に劣るとも、安いK_SOSを理由に総資源優位とも断定しない。

元分子への厳密保証、真のPF誤差、QPE/native・状態準備込みの総資源の優劣、新規性の確定を本決定から導かない。既存のSOURCE・raw・certificate・実行結果・科学契約・authorization・grant・ledger・消費済みclaim・STOPは全て保全する。

## ここで止める作業

追加分子・geometry・d/α/算術精度/trial走査、保存値の寄与診断、新保証探索・新SOURCE、SPEC-08、QPE/nativeへ進まない。古典証拠の安さを新しい主目的へ変える案も未採択であり、負の結果の事後的成功化や自動再開の理由にしない。

再開候補の具体的根拠と別途承認は [handoff](handoff.md) に固定した。今回の方針記録・公開は将来の科学実行許可ではない。**公開・保全確認後mandatory STOP。追加作業は不要であり、新しい根拠とユーザーの別途方針承認があるまで休止を維持する。**
