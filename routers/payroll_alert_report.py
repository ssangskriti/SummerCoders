import uvicorn
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdfkit
import logging
import base64
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import calendar
import jinja2

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed file path
FILE_PATH = "./data/Report_STD_TSD_QUERY_20240603_obfuscated.csv"

@router.get("/graphs/{graph_name}")
async def get_graph(graph_name: str):
    graph_path = f"graphs/{graph_name}"
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Graph not found")
    return FileResponse(graph_path)

@router.get("/")
async def main():
    content = """
    <body>
    <a href="/generate_pdf/" style="text-decoration: none;">
        <h4 style="display: inline-block; padding: 10px 20px; background-color: #000000; color: white; border-radius: 5px;">Payroll Report</h4>
    </a>
    </body>
    """
    return HTMLResponse(content=content)


# Function to process and clean the data
def process_data(file_path):
    try:
        logger.info("Reading CSV file")
        df = pd.read_csv(file_path)
        # Drop unnecessary columns
        df = df.drop(columns=['Unnamed: 9', 'Unnamed: 11', 'Unnamed: 13'])

        logger.info("Converting 'Date', 'Start Time', and 'End Time' to datetime")
        df['Date'] = pd.to_datetime(df['Date'])
        

        df['Start Time'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Start Time'])
        df['End Time'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['End Time'])

        logger.info("Adjusting 'End Time' for entries that end after midnight")
        df.loc[df['End Time'] < df['Start Time'], 'End Time'] += pd.Timedelta(days=1)

        logger.info("Calculating the duration")
        df['Duration'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 3600
        df['Duration'] = df['Duration'].round(2)

        logger.info("Adding 'Week' column")
        df['Week'] = df['Date'].dt.isocalendar().week

        df = df[~df['Pay Code'].str.contains('End Date: ...|Created by: ...', na=False)]

        pay_code_counts = df['Pay Code'].value_counts().reset_index()
        pay_code_counts.columns = ['Pay Code', 'Count']
        

        # Separate the holiday entries for the holiday table
        df_holidays = df[df['Pay Code'] == 'Holiday'].copy()

        # Remove rows where 'Pay Code' is 'Holiday' for other tables and graphs
        df_non_holidays = df[df['Pay Code'] != 'Holiday'].copy()

        return df_non_holidays, df_holidays, pay_code_counts
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail="Error processing data")



# filter for excessive and long hours
def filter_data(df):
    df_long_sessions = df[df['Duration'] > 5]
    df['Week'] = df['Date'].dt.isocalendar().week
    weekly_hours = df.groupby(['Name', 'Week'])['Duration'].sum().reset_index()
    df_excessive_hours = weekly_hours[weekly_hours['Duration'] > 10]
    df_filtered = df.merge(df_excessive_hours[['Name', 'Week']], on=['Name', 'Week'], how='inner')
    df_filtered = pd.concat([df_filtered, df_long_sessions]).drop_duplicates()
    return df_filtered 


def create_tables(df):
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week
    latest_year = df['Year'].max()
    latest_week = df[df['Year'] == latest_year]['Week'].max()
    previous_week = df[(df['Year'] == latest_year) & (df['Week'] < latest_week)]['Week'].max()
    
    # Last week
    df_current_week = df[(df['Year'] == latest_year) & (df['Week'] == previous_week) & (df['Duration'] > 10)]
    df_current_week['Duration'] = df_current_week['Duration'].round(2)
    start_date_last_week = df_current_week['Date'].min().strftime(' %m-%d-%Y ')
    end_date_last_week = df_current_week['Date'].max().strftime(' %m-%d-%Y ')
    table_current_week = df_current_week[['Name', 'Speed Type', 'Account', 'Duration']].rename(columns={'Speed Type': 'Code', 'Duration': 'Total Time (hours)'})
    
  
    
    return table_current_week,  start_date_last_week, end_date_last_week



def holiday_table(df):
    latest_date = df['Date'].max()
    ten_weeks_ago = latest_date - pd.Timedelta(weeks=10)
    df_holidays = df[(df['Pay Code'] == 'Holiday') & (df['Date'] >= ten_weeks_ago)]
    
    if not df_holidays.empty: 
        table_holidays = df_holidays[['Name', 'Date', 'Account', 'Speed Type', 'Value']].rename(columns={'Speed Type': 'Code', 'Value': 'Duration'})
        table_holidays['Date'] = table_holidays['Date'].dt.strftime(' %m-%d-%Y ')

        return table_holidays, ten_weeks_ago, latest_date
    else:
        return None, ten_weeks_ago, latest_date

    

def missing_time_table(df):
    latest_date = df['Date'].max()
    last_week = df[df['Date'] >= latest_date - pd.Timedelta(weeks=1)]
    df_missing_time = last_week[last_week['Start Time'].isnull() | last_week['End Time'].isnull()]
    if not df_missing_time.empty:
        table_missing_time = df_missing_time[['Name',   'Speed Type','Date', 'Start Time', 'End Time']].rename(columns={'Speed Type': 'Code'})
        table_missing_time['Date'] = pd.to_datetime(table_missing_time['Date']).dt.strftime(' %m-%d-%Y ')
        table_missing_time['Start Time'] = table_missing_time['Start Time'].dt.strftime('%H:%M').fillna('--')
        table_missing_time['End Time'] = table_missing_time['End Time'].dt.strftime('%H:%M').fillna('--')
        table_missing_time = table_missing_time.sort_values(by='Date')
        return table_missing_time
    else:
        return None

def extract_report_date(file_path):
    # Extract the correct part of the file name where the date is located
    file_name = os.path.basename(file_path)
    
    # Find the part of the file name that matches the date format (YYYYMMDD)
    date_str = None
    for part in file_name.split('_'):
        if part.isdigit() and len(part) == 8:
            date_str = part
            break
    
    if date_str is None:
        raise ValueError(f"No valid date found in file name {file_name}")
    
    # Convert the date string to the desired format
    report_date = datetime.strptime(date_str, "%Y%m%d").strftime("  %m-%d-%Y  ")
    return report_date

# overtime table
def get_student_ot_last_week(df, latest_date):
    # Calculate the start date of the last week
    start_of_last_week = latest_date - timedelta(days=7)

    # Filter the DataFrame for "Student OT" data in the last week
    student_ot_df = df[(df['Pay Code'] == 'Student OT') & (df['Date'] >= start_of_last_week) & (df['Date'] <= latest_date)]

    # Calculate the number of rows and total hours
    student_ot_rows = len(student_ot_df)
    student_ot_total_hours = student_ot_df['Duration'].sum()

    return student_ot_df, student_ot_rows, student_ot_total_hours



def generate_graphs(df):
    try:
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        latest_date = df['Date'].max()
        # Find the most recent Monday (start of the current week)
        most_recent_monday = latest_date - timedelta(days=latest_date.weekday())
        # Start of the 9th week before the current week
        start_of_tenth_week = most_recent_monday - timedelta(weeks=9)

        # List of all week start dates (Mondays) for the last 10 weeks
        all_week_starts = pd.date_range(start=start_of_tenth_week, end=most_recent_monday, freq='W-MON')

        # Filter the data for the last 10 weeks
        df_last_ten_weeks = df[(df['Date'] >= start_of_tenth_week) & (df['Date'] <= latest_date)]

        start_date_last_ten_weeks = start_of_tenth_week.strftime(' %m-%d-%Y ')
        end_date_last_ten_weeks = latest_date.strftime(' %m-%d-%Y ')

        # Get unique student names for color mapping
        students_plotted_ten_weeks = 0
        colors = plt.cm.get_cmap('tab20', df_last_ten_weeks['Name'].nunique())


        plt.figure(figsize=(14, 8))

        for i, (user, user_data) in enumerate(df_last_ten_weeks.groupby('Name')):
            account = user_data['Account'].iloc[0]
            speed_type = user_data['Speed Type'].iloc[0]
            legend_label = f"{user} ({account}, {speed_type})"
            user_data['Week_Start'] = user_data['Date'] - pd.to_timedelta(user_data['Date'].dt.weekday, unit='d')
            weekly_data = user_data.groupby('Week_Start')['Duration'].sum()
            excessive_hours = weekly_data[weekly_data > 10]

            if not excessive_hours.empty:
                students_plotted_ten_weeks += 1
                plt.plot(
                    excessive_hours.index, 
                    excessive_hours.values, 
                    marker='o', 
                    linestyle='-', 
                    color=colors(i), 
                    label=legend_label
                )
                plt.scatter(
                    excessive_hours.index, 
                    excessive_hours.values, 
                    color=colors(i), 
                    marker='*', 
                    s=150
                )
      
        plt.title(f'Excessive Weekly Hours (>10 hours/week) - Last 10 Weeks ({start_date_last_ten_weeks} to {end_date_last_ten_weeks})\n\n  {students_plotted_ten_weeks} students have been reported\n ', fontsize=20)
        plt.xlabel('First day of the week', fontsize=16)
        plt.ylabel('Duration (Hours)', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)

        # Ensure dates on the x-axis are in order and formatted correctly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(' %m-%d-%Y '))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=calendar.MONDAY))
        plt.gcf().autofmt_xdate()  # Auto-rotate date labels for better readability

        # Set the x-ticks to match the start of each week
        plt.xticks(all_week_starts, [date.strftime(' %m-%d-%Y ') for date in all_week_starts])

        plt.tight_layout()
        graph_path_excessive_hours = os.path.join(output_dir, 'excessive_hours.png')
        plt.savefig(graph_path_excessive_hours, bbox_inches='tight')
        plt.close()

        two_weeks_ago = latest_date - timedelta(weeks=2)
        df_last_two_weeks = df[df['Date'] >= two_weeks_ago]

        start_date_last_two_weeks = two_weeks_ago.strftime(' %m-%d-%Y ')
        end_date_last_two_weeks = latest_date.strftime(' %m-%d-%Y ')

        students_plotted_two_weeks = 0
        colors = plt.cm.get_cmap('tab20', df_last_two_weeks['Name'].nunique())


        plt.figure(figsize=(14, 8))
        for i, (user, user_data) in enumerate(df_last_two_weeks.groupby('Name')):
            account = user_data['Account'].iloc[0]
            speed_type = user_data['Speed Type'].iloc[0]
            legend_label = f"{user} ({account}, {speed_type})"
            long_sessions = user_data[user_data['Duration'] > 5]

            if not long_sessions.empty:
                students_plotted_two_weeks += 1
                plt.plot(user_data['Date'], user_data['Duration'], marker='o', linestyle='-', color=colors(i), label=legend_label)
                plt.scatter(long_sessions['Date'], long_sessions['Duration'], color=colors(i), marker='*', s=150)

        plt.title(f'Long Sessions (>5 hours/day) - Last 2 weeks ({start_date_last_two_weeks} to {end_date_last_two_weeks})\n\n{students_plotted_two_weeks} students have been reported\n', fontsize=20)
        plt.xlabel('Dates', fontsize=16)
        plt.ylabel('Duration (Hours)', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(' %m-%d-%Y '))
        plt.tight_layout()
        graph_path_long_sessions = os.path.join(output_dir, 'long_sessions.png')
        plt.savefig(graph_path_long_sessions, bbox_inches='tight')
        plt.close()

        return [graph_path_excessive_hours, graph_path_long_sessions]
    except Exception as e:
        logger.error(f"Error generating graphs: {e}")
        raise HTTPException(status_code=500, detail="Error generating graphs")




@router.get("/generate_pdf/")
async def generate_pdf():
    try:
        df_non_holidays, df_holidays, pay_code_counts = process_data(FILE_PATH)
        df_filtered = filter_data(df_non_holidays)
        graph_path_excessive_hours, graph_path_long_sessions = generate_graphs(df_filtered)
        table_current_week, start_date_last_week, end_date_last_week = create_tables(df_non_holidays)
        table_holidays, holidays_start_date, holidays_end_date = holiday_table(df_holidays)  
        missing_time = missing_time_table(df_non_holidays)
        missing_time_rows = len(missing_time) if missing_time is not None else 0

        report_date = extract_report_date(FILE_PATH)

        # table summaries
        current_week_rows = len(table_current_week) if table_current_week is not None else 0
        current_week_total_hours = table_current_week['Total Time (hours)'].sum() if table_current_week is not None else 0

        holiday_rows = len(table_holidays) if table_holidays is not None else 0
        holiday_total_hours = table_holidays['Duration'].sum() if table_holidays is not None else 0

        missing_time_rows = len(missing_time) if missing_time is not None else 0

        # Meta info
        total_rows_processed = len(df_non_holidays) + len(df_holidays)
        report_generation_datetime = datetime.now()
        report_generation_time = report_generation_datetime.strftime('%I:%M %p')
        report_generation_date = report_generation_datetime.strftime('%B %d, %Y')

        # Overtime info
        latest_date = df_non_holidays['Date'].max() 
        student_ot_df, student_ot_rows, student_ot_total_hours = get_student_ot_last_week(df_non_holidays, latest_date)
        
        missing_start_date = pd.to_datetime(missing_time['Date']).min().strftime(' %m-%d-%Y ')
        missing_end_date = pd.to_datetime(missing_time['Date']).max().strftime(' %m-%d-%Y ')
            
        holidays_start_date = holidays_start_date.strftime(' %m-%d-%Y ')
        holidays_end_date = holidays_end_date.strftime(' %m-%d-%Y ')

        # Load the HTML template from the file
        template_loader = jinja2.FileSystemLoader(searchpath="./")
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("./templates/payroll_reports/payroll_alert_report.html")

        # Render the template with the actual data, ensuring default values if None
        html_content = template.render(
            pay_code_counts=pay_code_counts.to_html(index=False, classes="table", header="true") if not pay_code_counts.empty else "",
            base64_string_excessive_hours=base64.b64encode(open(graph_path_excessive_hours, "rb").read()).decode('utf-8') if graph_path_excessive_hours else "",
            base64_string_long_sessions=base64.b64encode(open(graph_path_long_sessions, "rb").read()).decode('utf-8') if graph_path_long_sessions else "",
            current_week_rows=current_week_rows,
            start_date_last_week=start_date_last_week if start_date_last_week else "N/A",
            end_date_last_week=end_date_last_week if end_date_last_week else "N/A",
            current_week_total_hours=current_week_total_hours,
            table_current_week=table_current_week.to_html(index=False, classes="table", header="true") if not table_current_week.empty else "",
            missing_time_rows=missing_time_rows,
            missing_time_table=missing_time.to_html(index=False, classes="table", header="true") if missing_time is not None else "",
            missing_start_date=missing_start_date,
            missing_end_date=missing_end_date,
            holiday_rows=holiday_rows,
            holiday_total_hours=holiday_total_hours,
            holidays_start_date=holidays_start_date,
            holidays_end_date=holidays_end_date,
            holiday_table=table_holidays.to_html(index=False, classes="table", header="true") if table_holidays is not None else "",
            student_ot_rows=student_ot_rows,
            student_ot_total_hours=student_ot_total_hours,
            student_ot_table=student_ot_df.to_html(index=False, classes="table", header="true") if not student_ot_df.empty else "",
            report_date=report_date,
            report_generation_time=report_generation_time,
            report_generation_date=report_generation_date,
            file_name=os.path.basename(FILE_PATH),
            total_rows_processed=total_rows_processed
            
          )

        # Save the rendered HTML content to a file
        html_file = "graphs/payroll_report.html"
        with open(html_file, "w") as f:
            f.write(html_content)
        logger.info(f"HTML content saved to {html_file}")

        # Convert HTML to PDF using pdfkit
        try:
            pdf_file = "graphs/payroll_report.pdf"
            pdfkit.from_file(html_file, pdf_file)
            logger.info(f"PDF generated at {pdf_file}")
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise HTTPException(status_code=500, detail="Error generating PDF")

        return FileResponse(pdf_file, media_type='application/pdf', filename='payroll_report.pdf')
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        raise HTTPException(status_code=500, detail="Error handling file upload")


uvicorn.run(router, host="0.0.0.0", port=8005)