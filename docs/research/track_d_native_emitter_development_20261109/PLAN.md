# Native emitter・1Q界面一括開発：固定scope

2026-11-09 JST。基点b49ee3fc、同じリソース評価branchを継続する。既存SOURCE・結果・STOPを変更せず、新namespaceで許可された技術開発を行う。新P0・H4実行・保存H4再解析・sampling・fit・N／全RPE会計・詳細科学レビューは0。

1. 既存C3 mixed_head／Reference gadget／Hadamard wrapperを、元moduleをpatchせずconstructorを局所bindしたfunctionからNativeSinkへ接続する。既存のローカルimport型emitter（tail等）は小さいstep/chunkのまま再利用する。全C3 builder／sampler／Gaussian合成器を再実装しない。
2. Standard gate definitionを再帰展開し、fundamental UだけQiskit公式1Q Euler ZSXXでnative化する。global phase・actual inverse・mappingを保持。新full transpileは呼ばない。
3. 既存T0／CX bridgeを使い、Qiskit1.3の2×2 run合成だけを直接再利用し、native runが厳密に短くなる置換とCX相殺を繰り返す。一般1Q optimizerやP0全体を再実装しない。P0全域は未資格のままとする。
4. C7-2保存54人工caseの固定indexを使う。保存raw QPYが取得できるcaseをhash照合し、native化・partial bridge・位相付き小規模Operator・保存P0価格を比較する。最大5qubitの人工原本のみ。取得不能はHOLD、原本再生成なし。H4原本を開かない。
5. 新10人工testsと既存の重複しない37回帰を実行する。全wrapper/dense targetを持たないnative経路と、小さいQCを使うoracle／chunk準備を区別して記録する。q1024〜4096の新benchmarkは行わない。

component abs1e−10／rtol0／phase fitなしを維持する。未対応／P0不一致を隠さず、既存反例・主A／比較F/S・208−224qの未証明状態を保持する。出力はartifacts/track_d_native_emitter_development/2026-11-09/run01。code/PLANをcommitしてSOURCEを記録し、公開SHA/tree/bytes照合後mandatory STOPする。
