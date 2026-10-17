import unittest,copy
from cricket.cricsheet import convert_match,SkipMatch

def fixture():
    return {'info':{'dates':['2008-04-18'],'season':'2007/08','balls_per_over':6,'overs':20,'match_type':'T20','teams':['A','B'],'toss':{'winner':'A','decision':'bat'},'outcome':{'winner':'B'},'venue':'Ground'},'innings':[{'team':'A','overs':[{'over':0,'deliveries':[{'runs':{'total':4},'extras':{}},{'runs':{'total':0},'wickets':[{'kind':'retired hurt'}]}]}]},{'team':'B','target':{'overs':20,'runs':5},'overs':[{'over':0,'deliveries':[{'runs':{'total':1},'extras':{'wides':1}},{'runs':{'total':4}}]}]}]}

class ConverterTests(unittest.TestCase):
    def test_season_uses_match_year(self):
        rows=convert_match(fixture(),'123',strict_complete=False)
        self.assertEqual(rows[0]['season'],2008)
        self.assertEqual(rows[0]['source_season'],'2007/08')
    def test_extras_and_retirement(self):
        rows=convert_match(fixture(),'123',strict_complete=False)
        self.assertEqual(rows[1]['wicket'],0);self.assertEqual(rows[2]['legal'],0)
    def test_reject_dls(self):
        d=fixture();d['info']['outcome']['method']='D/L'
        with self.assertRaises(SkipMatch): convert_match(d,'1',strict_complete=False)
    def test_reject_shortened_target(self):
        d=fixture();d['innings'][1]['target']['overs']=15
        with self.assertRaises(SkipMatch): convert_match(d,'1',strict_complete=False)
    def test_reject_tie(self):
        d=fixture();d['info']['outcome']['result']='tie'
        with self.assertRaises(SkipMatch): convert_match(d,'1',strict_complete=False)
    def test_reject_incomplete_first_innings(self):
        with self.assertRaises(SkipMatch): convert_match(fixture(),'1')
    def test_reject_wrong_winner(self):
        d=fixture();d['info']['outcome']['winner']='A'
        with self.assertRaises(SkipMatch): convert_match(d,'1',strict_complete=False)
    def test_no_ball_with_runs_and_wicket(self):
        d=fixture();d['innings'][1]['overs'][0]['deliveries'][0].update(extras={'noballs':1},wickets=[{'kind':'run out'}])
        rows=convert_match(d,'1',strict_complete=False)
        self.assertEqual((rows[2]['total_runs'],rows[2]['wicket'],rows[2]['legal']),(1,1,0))
