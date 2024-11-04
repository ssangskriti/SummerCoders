from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime, timedelta
import pdfkit
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np

router = APIRouter()

# Load the CSV data
purpose_codes_df = pd.read_csv('./data/Purpose_Codes_All.csv')
report_std_tsd_df = pd.read_csv('./data/Report_STD_TSD_QUERY_20240603_obfuscated.csv')

templates = Jinja2Templates(directory="./templates/payroll_reports")

# Ensure all values in the 'Email' column are strings and handle NaN values
purpose_codes_df['Email'] = purpose_codes_df['Email'].fillna('').astype(str)
report_std_tsd_df['Date'] = pd.to_datetime(report_std_tsd_df['Date'])

# Helper function to calculate time difference considering midnight cross
def calculate_time_difference(start_time, end_time):
    try:
        start_dt = datetime.strptime(str(start_time), "%I:%M %p")
        end_dt = datetime.strptime(str(end_time), "%I:%M %p")
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        return (end_dt - start_dt).seconds / 3600  # Convert seconds to hours
    except ValueError:
        return None

# Helper function to get the latest week's data
def get_latest_week_data(df, weeks=1):
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    start_of_period = latest_date - timedelta(days=7 * weeks - 1)
    end_of_period = latest_date
    return df[(df['Date'] >= start_of_period) & (df['Date'] <= end_of_period)], start_of_period, end_of_period

@router.get("/", response_class=HTMLResponse)
async def get_email_form(request: Request):
    return templates.TemplateResponse("email_form.html", {"request": request})

@router.post("/faculty", response_class=HTMLResponse)
async def redirect_to_faculty_data(
    email: str = Form(...), 
    purpose_codes: str = Form(...), 
    start_date: str = Form(...), 
    end_date: str = Form(...)
):
    username = email.split('@')[0]
    purpose_codes_list = [code.strip() for code in purpose_codes.split(',')]
    return RedirectResponse(
        url=f"/faculty/{username}?purpose_codes={','.join(purpose_codes_list)}&start_date={start_date}&end_date={end_date}",
        status_code=303
    )

@router.get("/faculty/{username}", response_class=HTMLResponse)
async def get_faculty_data(request: Request, username: str, purpose_codes: str, start_date: str, end_date: str):
    # faculty_data = purpose_codes_df[purpose_codes_df['Email'].apply(lambda x: x.split('@')[0] if '@' in x else '') == username]
    # if faculty_data.empty:
    #     raise HTTPException(status_code=404, detail="Faculty not found")

    purpose_codes_list = purpose_codes.split(',')
    start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d")

    faculty_data = purpose_codes_df[purpose_codes_df['Email'].apply(lambda x: x.split('@')[0] if '@' in x else '') == username]
    if faculty_data.empty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    # Filter the report data by the provided date range and purpose codes
    filtered_report_std_tsd_df = report_std_tsd_df[
        (report_std_tsd_df['Date'] >= start_date_parsed) &
        (report_std_tsd_df['Date'] <= end_date_parsed) &
        (report_std_tsd_df['Speed Type'].isin(purpose_codes_list))
    ]

    response_data_week = []
    total_hours_week = 0

    latest_week_data, start_of_week, end_of_week = get_latest_week_data(report_std_tsd_df, weeks=1)
    
    for _, row in faculty_data.iterrows():
        purpose_code = row['Purpose Code']
        matching_records = latest_week_data[latest_week_data['Speed Type'] == purpose_code]
        
        for _, record in matching_records.iterrows():
            start_time = record['Start Time']
            end_time = record['End Time']
            hours = calculate_time_difference(start_time, end_time)
            if hours is None:
                hours_display = "Missing clock time"
                hours = 0
            else:
                hours_display = f"{hours:.2f}"
            total_hours_week += hours
            
            response_data_week.append({
                "Name": record['Name'],
                "Account": record['Account'],
                "Date": record['Date'].strftime("%m-%d-%Y"),
                "Purpose Code": purpose_code,
                "Hours": hours_display
            })
    
    response_data_week = sorted(response_data_week, key=lambda x: x['Date'])

    # Calculate total hours for each purpose code
    total_hours_per_purpose = {}
    for item in response_data_week:
        purpose_code = item['Purpose Code']
        if purpose_code not in total_hours_per_purpose:
            total_hours_per_purpose[purpose_code] = 0
        total_hours_per_purpose[purpose_code] += float(item['Hours']) if item['Hours'] != "Missing clock time" else 0
    total_hours_per_purpose = [{"Purpose Code": key, "Total Hours": f"{value:.2f}"} for key, value in total_hours_per_purpose.items()]

    response_data_10weeks = []
    total_hours_10weeks = 0

    last_10weeks_data, start_of_10weeks, end_of_10weeks = get_latest_week_data(report_std_tsd_df, weeks=10)
    
    for _, row in faculty_data.iterrows():
        purpose_code = row['Purpose Code']
        matching_records = last_10weeks_data[last_10weeks_data['Speed Type'] == purpose_code]
        
        for _, record in matching_records.iterrows():
            start_time = record['Start Time']
            end_time = record['End Time']
            hours = calculate_time_difference(start_time, end_time)
            if hours is None:
                hours_display = "Missing clock time"
                hours = 0
            else:
                hours_display = f"{hours:.2f}"
            total_hours_10weeks += hours
            
            response_data_10weeks.append({
                "Name": record['Name'],
                "Account": record['Account'],
                "Date": record['Date'].strftime('%m-%d-%Y'),
                "Purpose Code": purpose_code,
                "Hours": hours_display
            })
    
    response_data_10weeks = sorted(response_data_10weeks, key=lambda x: x['Date'])

    # Prepare data for the new table
    persons = sorted(set((item["Name"], item["Account"]) for item in response_data_10weeks))
    purpose_codes = sorted(set(item["Purpose Code"] for item in response_data_10weeks))
    table_data = {(person, account): {purpose_code: 0 for purpose_code in purpose_codes} for person, account in persons}

    for item in response_data_10weeks:
        if item["Hours"] != "Missing clock time":
            table_data[(item["Name"], item["Account"])][item["Purpose Code"]] += float(item["Hours"])

    totals_row = {purpose_code: sum(table_data[(person, account)][purpose_code] for person, account in persons) for purpose_code in purpose_codes}
    

    num_entries_week_table= len(response_data_week)
    
    num_entries_10weeks_table= len(response_data_10weeks)

     

    weekly_graph = generate_graph(response_data_week, 'Total Hours Spent by Each Student for Each Purpose Code  ', start_of_week, end_of_week,   num_entries_week_table)
    ten_weeks_graph = generate_graph(response_data_10weeks, 'Total Hours Spent by Each Student for Each Purpose Code ', start_of_10weeks, end_of_10weeks,  num_entries_10weeks_table)
    
    
    # Assume `students` is a set or list of all student names
    # Generate the bar plot for the last week
    # stacked_bar_plot = generate_stacked_bar_plot(
    #     latest_week_data,
    #     'Total Hours Spent by Each Student for Each Purpose Code Over Last Week'
    # )


    # Calculate total hours for each student and sort
    total_hours_per_person = [(person, account, sum(hours.values())) for (person, account), hours in table_data.items()]
    
    print_ten_weeks_graph = len(total_hours_per_person) <= 10

    if len(total_hours_per_person) > 10:
        top_10_students = sorted(total_hours_per_person, key=lambda x: x[2], reverse=True)[:10]
        bottom_10_students = sorted(total_hours_per_person, key=lambda x: x[2])[:10]
    else:
        top_10_students = []
        bottom_10_students = []

#  total hours for the last 10 weeks for each purpose code
    total_hours_per_purpose_10weeks = {}
    for item in response_data_10weeks:
        purpose_code = item['Purpose Code']
        if purpose_code not in total_hours_per_purpose_10weeks:
            total_hours_per_purpose_10weeks[purpose_code] = 0
        total_hours_per_purpose_10weeks[purpose_code] += float(item['Hours']) if item['Hours'] != "Missing clock time" else 0
    total_hours_per_purpose_10weeks = [{"Purpose Code": key, "Total Hours": f"{value:.2f}"} for key, value in total_hours_per_purpose_10weeks.items()]


      
    report_generation_time = datetime.now().strftime("%I:%M %p")
    report_generation_date = datetime.now().strftime("%m-%d-%Y")

    # Calculate total rows processed from both data files
    total_rows_processed = len(report_std_tsd_df)

    # Calculate total hours for the last week
    total_hours_for_week = sum(float(item['Total Hours']) for item in total_hours_per_purpose if item['Total Hours'] != "Missing clock time")

    # Calculate total hours for the last 10 weeks
    total_hours_for_10weeks = sum(float(item['Total Hours']) for item in total_hours_per_purpose_10weeks if item['Total Hours'] != "Missing clock time")


    # line plots

    # Generate line plot for the last week
    weekly_line_plot = generate_line_plot(
        latest_week_data,
        'Total Hours Per Student Over Last 7 Days',
        pd.date_range(start_of_week, end_of_week),
        students=set(latest_week_data['Name']),
        weekly_plot=True  # This is a weekly plot
    )

    # Generate line plot for the last 10 weeks
    ten_weeks_line_plot = generate_line_plot(
        last_10weeks_data,
        'Total Hours Per Student Over Last 10 Weeks ',
        pd.date_range(start_of_10weeks, end_of_10weeks, freq='W-MON'),
        students=set(last_10weeks_data['Name']),
        weekly_plot=False  # This is for the last 10 weeks
    )


    # return templates.TemplateResponse("research_faculty_data.html", {
    #     "request": request,
    #     "faculty_name": faculty_data.iloc[0]['Faculty Name'],
    #     "total_hours_week": f"{total_hours_week:.2f}" if total_hours_week > 0 else "Missing clock time",
    #     "details_week": response_data_week,
    #     "total_hours_per_purpose": total_hours_per_purpose,
    #     "start_of_week": start_of_week.strftime("%m-%d-%Y"),
    #     "end_of_week": end_of_week.strftime("%m-%d-%Y"),
    #     "table_data": table_data,
    #     "persons": persons,
    #     "purpose_codes": purpose_codes,
    #     "totals_row": totals_row,
    #     "weekly_graph": weekly_graph, 
    #     "print_ten_weeks_graph":print_ten_weeks_graph,
    #     "ten_weeks_graph": ten_weeks_graph,
    #     "total_hours_per_purpose_10weeks":total_hours_per_purpose_10weeks,
    #     "top_10_students": top_10_students,
    #     "bottom_10_students": bottom_10_students,
    #     "start_of_10weeks": start_of_10weeks.strftime("%m-%d-%Y"),
    #     "end_of_10weeks": end_of_10weeks.strftime("%m-%d-%Y"),
    #     "num_students_week": len(set(item['Name'] for item in response_data_week)),
    #     "num_entries_week_table":num_entries_week_table,
    #     "num_students_10weeks": len(set(item['Name'] for item in response_data_10weeks)),
    #     "num_purpose_codes_week": len(total_hours_per_purpose),
    #     "num_purpose_codes_10weeks": len(total_hours_per_purpose_10weeks),
    #     "report_generation_time": report_generation_time,
    #     "report_generation_date": report_generation_date,
    #     "file_name": 'Purpose Codes ALL.csv, Report_STD_TSD_QUERY_20240603_obfuscated.csv',
    #     "total_rows_processed": total_rows_processed,
    #     "total_hours_for_week": f"{total_hours_for_week:.2f}",
    #     "total_hours_for_10weeks": f"{total_hours_for_10weeks:.2f}",
    #     "weekly_line_plot": weekly_line_plot,
    #     "ten_weeks_line_plot": ten_weeks_line_plot,
    #     # "stacked_bar_plot":stacked_bar_plot,
    #     "num_entries_10weeks_table":num_entries_10weeks_table

    # })
 


    html_content = templates.TemplateResponse("research_faculty_data.html", {
        "request": request,
        "faculty_name": faculty_data.iloc[0]['Faculty Name'],
        "total_hours_week": f"{total_hours_week:.2f}" if total_hours_week > 0 else "Missing clock time",
        "details_week": response_data_week,
        "total_hours_per_purpose": total_hours_per_purpose,
        "start_of_week": start_of_week.strftime("%m-%d-%Y"),
        "end_of_week": end_of_week.strftime("%m-%d-%Y"),
        "table_data": table_data,
        "persons": persons,
        "purpose_codes": purpose_codes,
        "totals_row": totals_row,
        "weekly_graph": weekly_graph, 
        "print_ten_weeks_graph":print_ten_weeks_graph,
        "ten_weeks_graph": ten_weeks_graph,
        "total_hours_per_purpose_10weeks":total_hours_per_purpose_10weeks,
        "top_10_students": top_10_students,
        "bottom_10_students": bottom_10_students,
        "start_of_10weeks": start_of_10weeks.strftime("%m-%d-%Y"),
        "end_of_10weeks": end_of_10weeks.strftime("%m-%d-%Y"),
        "num_students_week": len(set(item['Name'] for item in response_data_week)),
        "num_entries_week_table":num_entries_week_table,
        "num_students_10weeks": len(set(item['Name'] for item in response_data_10weeks)),
        "num_purpose_codes_week": len(total_hours_per_purpose),
        "num_purpose_codes_10weeks": len(total_hours_per_purpose_10weeks),
        "report_generation_time": report_generation_time,
        "report_generation_date": report_generation_date,
        "file_name": 'Purpose Codes ALL.csv, Report_STD_TSD_QUERY_20240603_obfuscated.csv',
        "total_rows_processed": total_rows_processed,
        "total_hours_for_week": f"{total_hours_for_week:.2f}",
        "total_hours_for_10weeks": f"{total_hours_for_10weeks:.2f}",
        "weekly_line_plot": weekly_line_plot,
        "ten_weeks_line_plot": ten_weeks_line_plot

    }).body.decode("utf-8")
 
    pdfkit.from_string(html_content, 'faculty_report.pdf')
 
    return FileResponse('faculty_report.pdf', media_type='application/pdf', filename='faculty_report.pdf')


# Helper function to generate the bar plot for weekly and 10-week data
def generate_graph(data, title, start_date, end_date, entries):
    fig, ax = plt.subplots(figsize=(10, 6))
    purpose_codes = list(set(item['Purpose Code'] for item in data))
    students = list(set(item['Name'] for item in data))

    # Prepare data for stacked bar plot
    plot_data = {student: {purpose: 0 for purpose in purpose_codes} for student in students}
    for item in data:
        if item['Hours'] != "Missing clock time":
            plot_data[item['Name']][item['Purpose Code']] += float(item['Hours'])
    
    total_hours_per_student = {student: sum(hours.values()) for student, hours in plot_data.items()}
    sorted_students = sorted(students, key=lambda student: total_hours_per_student[student], reverse=True)

    # Plot data
    bottom = [0] * len(sorted_students)
    colors = plt.cm.tab20.colors  # Get a colormap with unique colors
    for i, purpose_code in enumerate(purpose_codes):
        heights = [plot_data[student][purpose_code] for student in sorted_students]
        bars = ax.bar(sorted_students, heights, bottom=bottom, label=purpose_code, color=colors[i % len(colors)])
        
        # Add text for each segment
        for bar, height, student in zip(bars, heights, sorted_students):
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y() - height / 2, f'{height:.2f}', ha='center', va='center', color='black', fontsize=8)
        
        bottom = [bottom[j] + heights[j] for j in range(len(bottom))]

    # Add total amount on top of each bar
    total_heights = [sum(plot_data[student].values()) for student in sorted_students]
    for bar, total in zip(bars, total_heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y(), f'{total:.2f}', ha='center', va='bottom', color='black', fontsize=10, weight='bold')

    ax.set_xlabel('Students')
    ax.set_ylabel('Hours')
    ax.set_title(f'{title}({start_date.strftime("%m-%d-%Y")} to {end_date.strftime("%m-%d-%Y")}) \n\n Number of students:  {entries}\n\n')
    ax.legend(loc='best')

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha="right")

    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return string



def generate_line_plot(data, title, date_range, students, weekly_plot=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check and calculate 'Hours' if not already present
    if 'Hours' not in data.columns:
        data['Hours'] = data.apply(lambda row: calculate_time_difference(row['Start Time'], row['End Time']), axis=1)

    # Aggregate data by student and date
    for student in students:
        student_data = data[data['Name'] == student]
        student_data = student_data.groupby('Date')['Hours'].sum().reset_index()

        # Plot the data
        if not student_data.empty:
            ax.plot(student_data['Date'], student_data['Hours'], label=student, marker='o')

    # Extract start and end dates from date_range
    # if isinstance(date_range, pd.DatetimeIndex) or isinstance(date_range, list):
    start_date = min(date_range).strftime('%m-%d-%Y')
    end_date = max(date_range).strftime('%m-%d-%Y')
    # else:
    #     raise ValueError("date_range should be a pd.DatetimeIndex or a list of datetime objects.")


    ax.set_xlabel('Date')
    ax.set_ylabel('Total Hours')
    ax.set_title(f'{title}({start_date} to {end_date})\n\n Number of Students: {len(students)}\n\n')
    
    if weekly_plot:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # Show only the first day of each week
    
    ax.grid(True)
    plt.xticks(rotation=45, ha="right")

    # Place the legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return string


def generate_stacked_bar_plot(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure 'Hours' column exists
    if 'Hours' not in data.columns:
        data['Hours'] = data.apply(lambda row: calculate_time_difference(row['Start Time'], row['End Time']), axis=1)

    # Aggregate data by student and purpose code
    students = data['Name'].unique()
    purpose_codes = data['Purpose Code'].unique()

    plot_data = pd.DataFrame(index=students, columns=purpose_codes).fillna(0)

    for student in students:
        student_data = data[data['Name'] == student]
        for purpose_code in purpose_codes:
            hours = student_data[student_data['Purpose Code'] == purpose_code]['Hours'].sum()
            plot_data.loc[student, purpose_code] = hours

    # Plot the data as a stacked bar plot
    bottom = np.zeros(len(students))
    colors = plt.cm.tab20.colors  # Get a colormap with unique colors

    for i, purpose_code in enumerate(purpose_codes):
        ax.bar(plot_data.index, plot_data[purpose_code], bottom=bottom, label=purpose_code, color=colors[i % len(colors)])
        bottom += plot_data[purpose_code]

    # Update the title to include the number of students
    title_with_details = f"{title} (Number of Students: {len(students)})"
    ax.set_xlabel('Students')
    ax.set_ylabel('Total Hours')
    ax.set_title(title_with_details)

    # Rotate the x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha="right")

    # Place the legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)

    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return string



uvicorn.run(router, host="0.0.0.0", port=8005)


 
