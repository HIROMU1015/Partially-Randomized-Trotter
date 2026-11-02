# CPU・wall・output会計

Total CPUはhost parent＋全worker、global wallはcaller outer（bridge前からdelivery close後まで）。pair CPU、IPC、observer、coarse/fineをtotalへ再加算せず、pair wallの和をglobal wallにしない。最後のcaller観測文書自身の保存とoffline consumerは後処理windowであり、保存byte creditは初めからglobal resultへ含める。

条件別parent/worker CPUとwallはS/P1の非重複window。共有setup、claim、receipt/diagnostic/STOP/finalization等の未配賦分はsharedに残す。条件別byteはparent traceのconditionとshared creditから導出する。S/P1のcapを二つ足す方式ではない。

pair記録はconstructor/source checks、generator+audit、angle/requery、evidence、serialization/hash/write/flush/finalize、coarse/fine代替partition、RPC、RSS/HWM/IO、retained bytesを保持する。親保存の複合診断は保存したが、親trace/state/全個別I/Oのexclusive費用やisolated observer overheadはnull。RSSのshared-page sumをunique厳密peakと呼ばない。失敗prefix・drain・unknown費用は正常比較へ混ぜない。

今回n256人工観測はparent 8.123531s、全worker 25.152102s、total 33.275633s、global wall 20.433934s。同一共有サーバーの人工形状の記述値で、実G7のCPU/保存量予測やspeedup主張ではない。

S→P1の順序、cold/warm cache、負荷、runtimeの交絡が残る。両者fusedなのでfused単独改善率は識別不能。固定portfolio運用費用の候補であり、machine-independent algorithm costやCPU母平均CIではない。量子signal confidenceをCPU CIへ転用しない。

旧人工trace41,899,756Bは特定形状の参考値に留める。今回のglobal trace/outputにも実科学上界の証明はない。Full-N31,710,848 callsのper-trial保存を実行可能と仮定しない。必要ならcheckpoint/transition集約と保持規則を別契約で設計する。
