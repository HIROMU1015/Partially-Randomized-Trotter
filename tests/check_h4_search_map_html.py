"""Presentation-only fixture: no scientific inputs or producer execution."""
from copy import deepcopy
import json
import re
import unittest
from trottertracks.h4_search_map_html import render, source_url


class SearchMapHTML(unittest.TestCase):
    def test_range_view_preserves_close_kappa_values_and_missing_records(self):
        points = [{'method':'PR_PAULI', 'id':str(i), 'kappa':{'kind':'PAPER_TOTAL_ROTATIONS',
                   'value':{'numerator':2**54+i, 'denominator':3}}} for i in (0, 1)]
        page = render({'points':points, 'targets':[{'kappa':None}, {'kappa':'2'}]})
        compact = json.loads(re.search(r'<script id="kappa-data" type="application/json">(.*?)</script>', page).group(1))
        self.assertNotEqual(*compact['points'])
        self.assertIsNone(compact['targets'][0])
        self.assertIn('登録候補にある値', page)
        self.assertIn('Q3評価で確認した値', page)

    def test_overview_links_the_exact_admitted_minimum_inside_the_all_runs_view(self):
        def cell(point, cost):
            return {'method':'PR_PAULI', 'row_label':'L_D=5', 'width_label':'1h',
                    'min_point_ids':[point], 'min_cost':{'numerator':cost, 'denominator':1}}
        cost = 2**54
        page = render({'views':[
            {'mode':'RUN', 'cells':[cell('other-run', 1)]},
            {'mode':'ALL', 'cells':[cell('more-expensive', cost + 1), cell('least', cost)]}],
            'points':[]})
        cards = page.split('<div class="results" id="overview">', 1)[1].split('</section>', 1)[0]
        self.assertEqual(re.findall(r'data-point="([^"]+)"', cards), ['least'])
        self.assertEqual(cards.count('価格未確定'), 3)
        self.assertIn(f'{cost:,}.00', cards)

    def test_offline_exact_data_source_trace_and_untrusted_labels(self):
        source={'commit':'a'*40,'path':'docs/research/space name/RESULT.json','selector':['rows',0]}
        index={'schema':'H4_SEARCH_MAP_V1','comparison':{'snapshot':'SYNTHETIC'},'runs':[],
            'points':[{'id':'point','method':'PR_DF','configuration_id':'cfg','formula':'2nd',
            'row_key':'ld1','row_label':'L_D=1','width_key':'1/5','width_label':'1/5',
            'requested_width':'</script><img src=x onerror=alert(1)>','N_X':[1]*13,'N_Y':[1]*13,
            'q3_status':'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','conditional_cost':{'numerator':11,'denominator':3},
            'sources':[source],'configuration_detail_key':'PR_DF:cfg'}],'configuration_details':{'PR_DF:cfg':{'schedule':[{'m':0,'q':1}], 'configuration_source':source}},'targets':[],'views':[{'id':'all','label':'全run累計','mode':'ALL','cells':[]},{'id':'new:run','label':'そのrunで新たに評価','mode':'NEW','run_id':'run','cells':[]}],
            'sources':[source],'issues':[]}
        page=render(index)
        embedded=re.search(r'<script id="index-data" type="application/json">(.*?)</script>',page,re.S).group(1)
        self.assertEqual(json.loads(embedded),index)
        self.assertNotIn('<img',page)
        self.assertNotIn('fetch(',page)
        self.assertNotIn('<script src=',page)
        self.assertIn('全収録点の静的表',page)
        self.assertIn('11',embedded)
        self.assertIn('3.67 RZ',page)
        self.assertIn('space%20name',page)
        self.assertIn('NEW',page)
        self.assertIn('そのrunで新たに評価',page)
        self.assertIn('SOURCE_NOT_IMPORTED',page)
        self.assertIn('p.schedule||cfg.schedule',page)
        self.assertIn('cfg.configuration_source',page)
        self.assertEqual(source_url(source),'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+'a'*40+'/docs/research/space%20name/RESULT.json')
        self.assertIsNone(source_url({'path':'local'}))
        # Presentation pages only; the source index and static fallback retain all 80.
        many=deepcopy(index);many['points']=[dict(index['points'][0],id='point'+str(i),configuration_id='cfg'+str(i))for i in range(80)]
        page80=render(many)
        embedded80=re.search(r'<script id="index-data" type="application/json">(.*?)</script>',page80,re.S).group(1)
        self.assertEqual(len(json.loads(embedded80)['points']),80)
        self.assertEqual(page80.count('<tr>'),81)
        self.assertIn('次の50件',page80)
        self.assertIn('ps.slice(shown,end)',page80)


if __name__=='__main__':unittest.main(verbosity=2)
