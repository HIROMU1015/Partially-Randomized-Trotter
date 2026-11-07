# 長round向け人工flat回路の段階メモリ検証

実H4は実行しない。専用branch `codex/track-a-b2-large-artificial-memory-20261107`、基点`96efc6123b8bea3ff66d5dea729e0dcc3b9908b4`。
旧SOURCE/raw/grant/marker/STOPと他worktreeを保全する。

- 既存人工controlled primitiveを短い3 qubit unitにし、既存compilerでrz/sx/x/cxのflat unitへ変換する。
  9 qubit（system8＋ancilla最後）上のsystem pairを巡回してcompose。H4 HamiltonianやRTE trajectoryではない。
- workloadは約4,096、65,536、262,144 instructionsの3段階（unit長の倍数へ切上げ）。
  これは実験条件でありCPU/RAMのcapではない。各条件は1観測のpilot。
- generationは全中間tailをlist保持する方式と、同じiteratorを1個ずつ消費する方式を独立processで比較。
  instruction digest・gate counts・global phaseで同一の完成flat回路を確認する。
- stream生成側でnative copy、X/Y wrapper、入力DAG、whole compile、compiled metrics/DAGを測る。
  旧/新の完全canonical fingerprintも同じnativeで比較するが、旧hashのallocator履歴がcompiler測定へ混ざらないよう最後に実施。
- wrapper、compiler engine、metric helper、fingerprintは既存実装を再利用。compiler probeではQiskit transpileを直接呼び、
  hashを分離する。偽のactual fingerprintを渡したcost recordは作らない。
- Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・logical all-to-allを維持。
- parentが自分のspawn workerの/proc RSSを5ms間隔で観測。begin/end handshake、実際の最大観測間隔、
  Python traced peak、前後RSS、kernel累積HWMを記録する。sampled peakは未観測の短いpeakに対する上限ではない。
  Python peakとRSSの単純差をRust/Cの厳密使用量とはしない。
- 時間比較を自worker同士の競合から分離するため条件は順次実行する。CPU/thread/RAM制限や共有設定は変更しない。
  次サイズのメモリ需要を直前の実測から見積もり実際の空き容量と比較し、実行中も空きRAM/cgroup headroomを観測。
  実際の残余メモリがそのstageの増加量を下回る場合などは自workerだけを安全停止する。PSI/CPU競合だけでは止めない。
- 完了・安全STOP後はJSON/CSV/REPORTを公開、remote SHA/tree/全変更ファイルを照合してmandatory STOP。
  実H4追加build/compile、価格MC、強PF価格、総RPE費用、新科学条件へ進まない。m17実H4の確定値へ外挿しない。
