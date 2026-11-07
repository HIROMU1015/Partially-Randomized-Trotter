# Track C改良PR：資源分解と追加改善

基点b9dacb639eabdd3cf499e89f8807ff50b68e23ef。固定linear H4/1Å/STO-3G/DF12/LD3、保存元H ground、18round/.02、energy1.5936e-4Ha、alpha.05、beta.4、compiler A、状態準備除外を維持する。RTE sampling law/return aggregation/他Trackは変更しない。旧SOURCE/raw/grant/ledger/marker/STOPには書き込まない。

まず前回のraw部品価格・native誤差予算を再利用して、D、RTE、wrapperとsignedなD/D境界補正を加算整合させる。D/R cut費を未取得の独立componentとして架空計上しない。basis・controlled由来の内訳は必要なlocal templateだけを測定し、isolated和と実際のcompileとの差をsigned correctionとして残す。depthはserial上界とDAG depthを区別する。

固定shotの成分削減/no-normalizationは仮想感度に限定する。実現可能な最初の改善は既存shot_rowsのcost-sensitive alphaを両armに適用し、保存済みnative rho/betaから新しい整数shotを区間算術で計算する。以後、分解で価値が確認されたD step/ground係数・境界の候補のみ限定検証する。SCI入力・RTE lawは固定のまま、合理的な同一仮説内で実装/修正/再検証を一括継続する。

C-N/C2-Aの短whole fitは長roundへ未資格だったため流用せず、前回の具体的modular実装の加算価格を再利用する。C2-Bの誤差配分と強baseline最適性未確定を踏まえ、局所・有限menu改善とglobal優越を分ける。既存の条件付き保存行列保証を継承/必要拡張し、元連続H包含を新しく主張しない。成果公開照合後mandatory STOP、詳細GPT科学レビューなし。
