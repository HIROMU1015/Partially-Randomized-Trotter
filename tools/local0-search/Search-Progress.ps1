function Get-RpeSearchDisplayStatus($Result) {
    if ($Result.runtime.phase -in @('FAILED','RECONCILIATION_REQUIRED','TECHNICAL_FAILURE') -or
        $Result.runtime.stop_reason -eq 'RUNTIME_EXCEPTION') {
        return 'TECHNICAL_FAILURE'
    }
    if ($Result.runtime.phase -eq 'COMPLETE' -and $Result.runtime.stop_reasons) {
        $holds = @('UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD')
        foreach ($reason in $Result.runtime.stop_reasons.PSObject.Properties) {
            if ($reason.Value -in $holds) { return 'TECHNICAL_FAILURE' }
        }
    }
    return $Result.status
}

function Get-RpeCandidateStateLabel([string]$State) {
    $labels = @{
        REUSED_SAVED_RESULT = 'Saved result reused'
        PREPARATION_WAITING = 'Waiting for preparation'
        PREPARATION_RUNNING = 'Preparing'
        EVALUATION_WAITING = 'Waiting for evaluation'
        EVALUATION_RUNNING = 'Evaluating'
        EVALUATION_COMPLETED = 'Evaluation completed'
        BUDGET_HOLD = 'Budget exhausted'
        UNSUPPORTED_GENERATION = 'Unsupported generation rule'
        MISSING_FIXED_INPUT = 'Missing fixed input'
        TECHNICAL_FAILURE = 'Technical failure'
    }
    if ($labels.ContainsKey($State)) { return $labels[$State] }
    return $State
}

function Write-RpeSearchProgress($Result) {
    $progress = $Result.progress
    if ($progress.schema -ne 'RPE_LOCAL0_PROGRESS_V1') {
        Write-Host 'Effective budgets, resources, and candidate progress were not supplied by RomuTail.'
        return
    }
    if ($progress.effective) {
        Write-Host ('Effective settings: '+($progress.effective | ConvertTo-Json -Compress -Depth 30))
    }
    foreach ($candidate in $progress.candidates) {
        $label = Get-RpeCandidateStateLabel $candidate.state
        $design = if ($candidate.design) { ' '+($candidate.design | ConvertTo-Json -Compress -Depth 12) } else { '' }
        $detail = if ($candidate.reason) { ' ('+[string]$candidate.reason+')' } else { '' }
        Write-Host ('Candidate '+$candidate.candidate_id+' '+$candidate.method+$design+' '+$label+$detail)
    }
    foreach ($stage in $progress.stages) {
        Write-Host ('Stage '+$stage.stage+' | queued/accepted: '+(Format-RpeProgressCount $stage.queue_accepted)+
            ' | execution started: '+(Format-RpeProgressCount $stage.execution_started)+
            ' | execution completed: '+(Format-RpeProgressCount $stage.execution_completed)+
            ' | execution unit: '+$stage.execution_unit)
    }
}

function Format-RpeProgressCount($Count) {
    if ($null -eq $Count) { return 'unknown' }
    return [string]$Count
}
