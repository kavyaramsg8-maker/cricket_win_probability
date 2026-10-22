"""Convert the original IPL JSON ZIP to the canonical project CSV."""
import argparse,json
from cricket.cricsheet import convert_zip

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True)
    p.add_argument('--output',default='data/raw/ipl_deliveries.csv')
    p.add_argument('--report',default='reports/data_conversion.json')
    p.add_argument('--min-year',type=int,default=2008)
    p.add_argument('--max-year',type=int,default=2024)
    a=p.parse_args();result=convert_zip(a.zip,a.output,a.report,min_year=a.min_year,max_year=a.max_year)
    print(json.dumps({k:v for k,v in result.items() if k!='excluded'},indent=2))
    print('Next: python run_pipeline.py --input '+a.output)

if __name__=='__main__': main()
