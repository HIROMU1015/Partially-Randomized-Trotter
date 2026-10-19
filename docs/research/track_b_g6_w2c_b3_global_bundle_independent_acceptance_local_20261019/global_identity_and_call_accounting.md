# Global identity・call会計

4方式×A/B×X/Y=16cell。各条件correctness16/cell=256、baseline256/cell=4096、instrumented256/cell=4096、計8448。二条件16896。独立旧plannerから48task/offsetを導出し、候補・manifest・control・実statefile・ordered trace・global receiptと照合した。

modeはtask key、global trial ID(run/mode/local trial)、phase0–5、offsetへ入る。local trialはphase/pass/arm/layer/axis/index。48 cursorと現在ticketで一意性を検査し、全trial ID集合を保持しない。正常6 barrier、二重RESERVE/COMPLETE、STOP後ENTER、phase違反、S失敗後P1開始を拒否する。

public n256と別fresh n256のplanned/reserved/started/completedは各16896。単条件projectionでも計画statefileは両条件1536の形状を保持し、実施768だけを正常conditionと認める。Coalesced RPC数とtrial数は区別。ENTERは親permitの観測点で、worker物理entry時刻の証明ではない。

[全case算術](global_call_accounting.json)を参照。科学NやFull-Nを正式採択・実行したものではない。
