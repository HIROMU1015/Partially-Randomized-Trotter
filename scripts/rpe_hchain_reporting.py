"""H-chain progress projection, leaving exact results and final exports intact."""


def intermediate_report(document):
    """Omit repeated evaluation arrays only while the optimizer is running.

    The durable original job outcomes and the selector's full in-memory
    results are unchanged. The normal final report retains the old schema
    and exports exact resume records.
    """
    projected=dict(document,report_phase='INTERMEDIATE',
                   resume_records_exported=False,
                   exact_result_source='jobs/<method>/<job_id>/OUTCOME.json')
    methods={}
    for method,result in document.get('methods',{}).items():
        row=dict(result)
        counts={}
        for name in ('evaluated','resume_records'):
            if isinstance(row.get(name),list):
                counts[name]=len(row.pop(name))
        if counts:
            row['intermediate_evaluation_counts']=counts
        methods[method]=row
    projected['methods']=methods
    return projected
