"""Person 1: convert Cricsheet nested JSON without extracting ZIP paths.
Specification: https://cricsheet.org/format/json/
Conservative filtering: record a reason for every excluded match.
"""
from pathlib import Path
from collections import Counter
import json,zipfile,hashlib
import pandas as pd

class SkipMatch(ValueError): pass

def convert_match(data,match_id,strict_complete=True):
    info=data['info'];innings=data.get('innings',[]);outcome=info.get('outcome',{})
    if info.get('match_type')!='T20' or info.get('overs')!=20 or info.get('balls_per_over',6)!=6:
        raise SkipMatch('nonstandard_format')
    if outcome.get('result') or not outcome.get('winner') or outcome.get('eliminator') or info.get('bowl_out'):
        raise SkipMatch('tie_no_result_or_eliminator')
    if outcome.get('method'): raise SkipMatch('adjusted_result_method')
    if len(innings)!=2 or any(i.get('super_over') for i in innings): raise SkipMatch('not_two_regulation_innings')
    # Missing player-of-match is harmless. Missing deliveries/innings are not.
    if any(x!='player_of_match' for x in info.get('missing',[])):
        raise SkipMatch('missing_match_data')
    teams=info['teams']
    if len(teams)!=2 or set(i['team'] for i in innings)!=set(teams): raise SkipMatch('invalid_teams')
    if any(i.get('penalty_runs') for i in innings): raise SkipMatch('innings_penalty_runs')
    if any(i.get('miscounted_overs') for i in innings): raise SkipMatch('miscounted_overs')
    if any(i.get('declared') or i.get('forfeited') or i.get('absent_hurt') for i in innings):
        raise SkipMatch('unusual_innings_ending')
    date=str(info['dates'][0]);year=int(date[:4]);source_season=str(info['season'])
    # IPL seasons such as 2007/08 actually occurred in calendar 2008.
    rows=[];summaries=[]
    for number,inn in enumerate(innings,1):
        target=inn.get('target',{})
        if target and float(target.get('overs',20))!=20: raise SkipMatch('shortened_target')
        total=legal=wickets=delivery_index=0
        for over in inn['overs']:
            for d in over['deliveries']:
                if legal>=120 or wickets>=10: raise SkipMatch('delivery_after_innings_end')
                if number==2 and total>=summaries[0]['total']+1: raise SkipMatch('delivery_after_target')
                extras=d.get('extras',{});runs=d['runs']['total']
                wicket=sum(w['kind'] != 'retired hurt' for w in d.get('wickets',[]))
                # Retired out counts as a wicket. Retired hurt does not.
                if wicket>1: raise SkipMatch('multiple_wickets_on_delivery')
                is_legal=int(not(extras.get('wides',0) or extras.get('noballs',0)))
                if not is_legal and runs<1: raise SkipMatch('illegal_delivery_without_extra')
                delivery_index+=1
                rows.append(dict(match_id=str(match_id),season=year,source_season=source_season,date=date,
                    venue=info.get('venue','Unknown venue'),innings=number,delivery_index=delivery_index,
                    over=over['over'],batting_team=inn['team'],bowling_team=next(t for t in teams if t!=inn['team']),
                    total_runs=runs,wicket=wicket,legal=is_legal,winner=outcome['winner'],
                    toss_winner=info['toss']['winner'],toss_decision=info['toss']['decision']))
                total+=runs;legal+=is_legal;wickets+=wicket
        if legal>120 or wickets>10: raise SkipMatch('innings_exceeds_limits')
        summaries.append({'total':total,'legal':legal,'wickets':wickets})
    first,second=summaries;target=first['total']+1
    declared_target=innings[1].get('target',{}).get('runs',target)
    if declared_target!=target: raise SkipMatch('adjusted_target_runs')
    if second['total']==first['total']: raise SkipMatch('regulation_tie')
    expected=innings[1]['team'] if second['total']>=target else innings[0]['team']
    if outcome['winner']!=expected: raise SkipMatch('score_winner_mismatch')
    if strict_complete:
        if first['legal']!=120 and first['wickets']!=10: raise SkipMatch('incomplete_first_innings')
        if second['total']<target and second['legal']!=120 and second['wickets']!=10:
            raise SkipMatch('incomplete_second_innings')
    return rows

def convert_zip(source,output,report_path):
    source=Path(source);records=[];excluded=[];accepted=[]
    with zipfile.ZipFile(source) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX/'))
        for name in names:
            mid=Path(name).stem
            try:
                rows=convert_match(json.loads(z.read(name)),mid)
                records.extend(rows);accepted.append(mid)
            except SkipMatch as exc: excluded.append({'match_id':mid,'reason':str(exc)})
            except (KeyError,TypeError,ValueError) as exc:
                excluded.append({'match_id':mid,'reason':'invalid_schema','detail':str(exc)})
    if not records: raise ValueError('No supported matches found')
    df=pd.DataFrame(records).sort_values(['date','match_id','innings','delivery_index'],kind='stable')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);df.to_csv(output,index=False)
    report={'source_filename':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'json_matches':len(names),'included_matches':len(accepted),'excluded_matches':len(excluded),
            'delivery_rows':len(df),'years':sorted(int(x) for x in df.season.unique()),
            'exclusion_counts':dict(Counter(x['reason'] for x in excluded)),'excluded':excluded,
            'season_policy':'Calendar year of first match date; original season preserved in source_season',
            'source_url':'https://cricsheet.org/downloads/'}
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2))
    return report
