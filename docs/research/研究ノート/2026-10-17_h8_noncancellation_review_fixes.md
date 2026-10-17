# 2026-10-17 H8 非相殺bias診断：レビュー指摘2点の修正

Codexの読取レビュー（16:51 JST）が指摘した2点を、科学条件を変えずに修正した。新SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`、未承認plan v2 digest `002f9edbf35506fca256431df641552d7c319581f306f13a316b736e07897e6d`。
(1) PR保存paired費用の再現gateとn32 pairing検査は費用比較stageだけで実行されるため、結果に stage 付きの未実行記録を出し、plan・本文をstage別に訂正した。(2) S4 modelの「検証済み」来歴表示をB2全PASS結合時だけに限定し、それ以外は未検証と明示した。
修正前に失敗する人工回帰test2件を追加し、修正後PASS。人工56件（2 interpreter）＋回帰113件PASS。計算値は修正前後で同一。v2 draftは入力読取前に拒否された。
保護対象のworktreeを動かさないよう独立cloneで作業した。一方、レビュー原文の広域grepが並行B2 attempt01をMEMORY_PRESSUREでSTOPさせ、準備段階のworktree変更が別sessionのB2準備attempt01〜03をSTOPさせていたことが判明し、保全receipt v2に記録した。
実H8の非相殺bias・shot・費用・ランキング・新signal/compile/sampling/fit・B2再実行・有効grantは0。次はユーザーがshot診断stageの承認とB2の扱いを判断する。mandatory STOP。
