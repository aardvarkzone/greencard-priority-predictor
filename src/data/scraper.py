import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def parse_month_list(text):
   """Parse a column of text containing months, removing 'India' and any year headers or category headers"""
   # Split text into lines
   lines = [line.strip() for line in text.split('\n') if line.strip()]
   
   # Remove any lines that are years, contain 'India', or are category headers
   months = []
   category_headers = ['f-1', 'f-2a', 'f-2b', 'f-3', 'f-4']
   for line in lines:
       try:
           int(line)  # Skip if line is a year
           continue
       except ValueError:
           line_lower = line.lower()
           if (line_lower != 'india' and 
               not any(header in line_lower for header in category_headers)):
               months.append(line)
   
   return months

def scrape_india_visa_data(url, category_type):
   """
   Scrape visa bulletin movement for India (both employment and family based)
   """
   print(f"\n=== Starting India {category_type} Movement Scraping ===")
   
   headers = {
       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
       'Accept-Language': 'en-US,en;q=0.5',
       'Connection': 'keep-alive'
   }
   
   try:
       response = requests.get(url, headers=headers)
       response.raise_for_status()
       
       soup = BeautifulSoup(response.content, 'html.parser')
       tables = soup.find_all('table')
       
       all_data = []
       cutoff_date = pd.to_datetime('2015-09-01')
       
       for table in tables:
           rows = table.find_all('tr')
           for row in rows:
               cols = row.find_all('td')
               if len(cols) < 2:
                   continue
               
               first_col = cols[0].get_text(separator='\n').strip()
               
               # Check if this is a pre-2006 row
               if category_type == "Family-Based":
                   try:
                       # Try to find a year in the first line
                       first_line = first_col.split('\n')[0].strip()
                       year = int(first_line)
                       if 1990 <= year <= 2006:
                           # This is a pre-2006 row
                           months = parse_month_list(first_col)
                           
                           # Get data columns
                           f1_dates = parse_month_list(cols[1].get_text(separator='\n'))
                           f2a_dates = parse_month_list(cols[2].get_text(separator='\n'))
                           f2b_dates = parse_month_list(cols[3].get_text(separator='\n'))
                           f3_dates = parse_month_list(cols[4].get_text(separator='\n'))
                           f4_dates = parse_month_list(cols[5].get_text(separator='\n'))
                           
                           # Create entries for each month
                           for i, month in enumerate(months):
                               data = {
                                   'date': f"{month} {year}",
                                   'f1_india': f1_dates[i] if i < len(f1_dates) else None,
                                   'f2a_india': f2a_dates[i] if i < len(f2a_dates) else None,
                                   'f2b_india': f2b_dates[i] if i < len(f2b_dates) else None,
                                   'f3_india': f3_dates[i] if i < len(f3_dates) else None,
                                   'f4_india': f4_dates[i] if i < len(f4_dates) else None
                               }
                               all_data.append(data)
                               print(f"Processed pre-2006 row: {data['date']}")
                           continue
                   except (ValueError, IndexError):
                       pass
               
               # Regular processing for post-2006 data
               date_cell = first_col.split('•')[0].strip()
               date_cell = date_cell.split('early file')[0].strip()
               
               # Skip header rows
               if ('visa bulletin' in date_cell.lower() or 
                   'india' in date_cell.lower()):
                   continue
               
               try:
                   row_date = pd.to_datetime(date_cell, format='%B %Y')
               except:
                   continue

               # Get values based on whether we're before or after September 2015
               if row_date <= cutoff_date:
                   if category_type == "Employment-Based":
                       data = {
                           'date': date_cell,
                           'eb1_india': cols[1].text.strip(),
                           'eb2_india': cols[2].text.strip() if len(cols) > 2 else None,
                           'eb3_india': cols[3].text.strip() if len(cols) > 3 else None
                       }
                   else:
                       data = {
                           'date': date_cell,
                           'f1_india': cols[1].text.strip(),
                           'f2a_india': cols[2].text.strip() if len(cols) > 2 else None,
                           'f2b_india': cols[3].text.strip() if len(cols) > 3 else None,
                           'f3_india': cols[4].text.strip() if len(cols) > 4 else None,
                           'f4_india': cols[5].text.strip() if len(cols) > 5 else None
                       }
               else:
                   if category_type == "Employment-Based":
                       data = {
                           'date': date_cell,
                           'eb1_india': cols[1].text.strip().split('\n')[0].strip(),
                           'eb2_india': cols[2].text.strip().split('\n')[0].strip() if len(cols) > 2 else None,
                           'eb3_india': cols[3].text.strip().split('\n')[0].strip() if len(cols) > 3 else None
                       }
                   else:
                       data = {
                           'date': date_cell,
                           'f1_india': cols[1].text.strip().split('\n')[0].strip(),
                           'f2a_india': cols[2].text.strip().split('\n')[0].strip() if len(cols) > 2 else None,
                           'f2b_india': cols[3].text.strip().split('\n')[0].strip() if len(cols) > 3 else None,
                           'f3_india': cols[4].text.strip().split('\n')[0].strip() if len(cols) > 4 else None,
                           'f4_india': cols[5].text.strip().split('\n')[0].strip() if len(cols) > 5 else None
                       }
               
               all_data.append(data)
               print(f"Processed row: {date_cell}")
       
       if not all_data:
           return None
           
       df = pd.DataFrame(all_data)
       df['date'] = pd.to_datetime(df['date'], format='%B %Y')
       df = df.sort_values('date')
       
       print(f"\nCollected {len(df)} rows of data")
       return df
       
   except requests.exceptions.RequestException as e:
       print(f"Error fetching webpage: {e}")
       return None
   except Exception as e:
       print(f"Error processing data: {e}")
       print(f"Error details: {str(e)}")
       return None

if __name__ == "__main__":
   print("Starting India Visa Bulletin Scraper...")
   
   urls = {
       "Employment-Based": "https://www.jackson-hertogs.com/us-immigration/visa-bulletin-and-quota-movement-2/employment-based-quota-bulletin-movement-2-2/",
       "Family-Based": "https://www.jackson-hertogs.com/us-immigration/visa-bulletin-and-quota-movement-2/india-family-based-quota-bulletin-movement/"
   }
   
   try:
       for category_type, url in urls.items():
           df = scrape_india_visa_data(url, category_type)
           
           if df is not None and not df.empty:
               filename = f"india_{category_type.lower().replace('-', '_')}_movement.csv"
               df.to_csv(filename, index=False)
               
               print(f"\n{category_type} Data Summary:")
               print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
               print(f"Total Records: {len(df)}")
               print("\nMost recent entries:")
               print(df.nlargest(5, 'date'))
           else:
               print(f"No {category_type} data was collected")
               
   except Exception as e:
       print(f"Error in main execution: {e}")