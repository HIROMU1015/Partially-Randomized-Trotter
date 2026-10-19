# 実科学入口の閉鎖

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

実schemaの `run_ready/content_adopted/contract_grants_execution=false`、`science_release/scientific_adoption=null`。retryはint 0、resumeはbool false。`production_dispatch=false`。型も比較する。

static.pyの `production_entry`、`real_bits`、`real_authority` は無条件raiseだけで、science loader/callable dispatchは実装されていない。別processから各3種類の引数（空、run_ready/fake grant、偽release/adoption）を渡す計9呼出しは全て拒否。fake authorityやboolean反転で開く経路は確認されない。

親とspawn workerで6科学moduleのimport要求をMetaPathFinderでload前に拒否。実import後に判定する試験ではない。科学moduleがsys.modulesへ追加されないことを保存。独立offline/test processはProducer port/exercise/authority/self-checkerと科学moduleのguardを別に設置した。

Python guardは協調的な検証の境界であり、悪意ある任意Python実行に耐えるOS sandboxではない。SOURCE変更やguard迂回に対するsecurity保証を主張しない。

実G7/F3 import、science bits/sampling、new science calculation/reanalysis、本番correctness/pilot/Full-N/native quantum、実authorization/grant/claim、正式SOURCE採択は全て0。
