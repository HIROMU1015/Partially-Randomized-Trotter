# D1：固定sourceへのone-shot実行authorization receipt

2026-10-13 JST。固定source S：`874ca12d347f8f35e5d8d64f65f24abc4a2df523`。

利用者の明示メッセージ原文：

> 続きの作業を行って大丈夫です

直前に案内したD1 source preparationの次段階（一回のみの登録D1比較）の承認として受領した。
固定契約SHA256：`afa174cc7aba9ff8a14ae1a77886e50792daaf8be342ea6c1f2d3024a4e543b8`。
Sの直接子Aでauthorization JSONとこのreceiptだけを変更する。Aをpushしてremote/clean identityを確認後、固定observer/runnerを一回だけ起動する。

P5、既存provider、tau=5/7、r=1,2、六方式、partial対P5 primary、accuracy/confidence、native/error/zero-fill会計、runtime/tool identity、資源上限を変更しない。
workers=2、numeric threads=1、retry=0、全outcomeでmandatory STOP。
旧結果・source・authorization・marker・STOPは不変。追加grid、SP、DF、分子/NPZ、GPU、G11、RA-D0、Track A/N3へ進まない。
実行後は保存結果と外側process・来歴をread-only監査し、必要資料を公開してGPTの科学判断へ戻す。
