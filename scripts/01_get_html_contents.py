# This script scrapes the contents of the URLs provided in the Google Doc and 
# the URLs from the Farmer School of Business Bulletin. The contents are saved
# in multiple pickle files for future use.

import os
import requests
import re
import pickle

from bs4 import BeautifulSoup
import numpy as np

from langchain_community.document_loaders import PlaywrightURLLoader, SeleniumURLLoader
from urllib.parse import urljoin


# Bulletin URLs for the Farmer School of Business
# ------------------------------------------------------------------------------

# URL of the webpage
url = "https://bulletin.miamioh.edu/farmer-business/"

# Send a request to the webpage
response = requests.get(url)

# Parse the HTML content of the webpage
soup = BeautifulSoup(response.content, 'html.parser')

# Find all 'a' tags within the specified CSS selector
links = soup.select('#degreesandprogramstextcontainer > ul > li > a')

# Extract the href attribute and convert to absolute URL
absolute_urls = [urljoin(url, link['href']) for link in links]



# URLs from Chanelle's Google Doc (https://docs.google.com/document/d/1duX3J7KSIBiBhkuTKDic_dNcZoPgDlQQO_Oqrb9DroU/edit)
# ------------------------------------------------------------------------------
urls_from_google_doc = [
  # Student Appointments and Early Registration
  'https://miamioh.edu/fsb/student-resources/academic-advising/appointments.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/registration.html',
  
  # Majors
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/majors.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/accountancy.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/analytics.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/business-analytics.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/business-economics.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/entrepreneurship.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/finance.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/human-capital-management-leadership.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/information-and-cybersecurity-management.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/marketing.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/real-estate.html',
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/departments/supply-chain-and-operations-management.html',
  
  # Minors and Certificates
  'https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates1.html',
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/accountancy-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/arts-mgmt.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/business-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/ba-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/bgm-certificate.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/climacc-eng-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/creativity-esp.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/cybersec-mgmt.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/cyberacc-mgmt.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/deals-cert.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/eco-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/esp-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/fin-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/fnd-ba-cert.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/hcml-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/healthcare-sales-cert.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/is-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/ib-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/mgt-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/mkt-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/re-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/scm-minor.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/social-esp-cert.html",
  "https://miamioh.edu/fsb/student-resources/academic-advising/academics/minors-and-certificates/startup-esp-cert.html",
  
  # Business Core
  "https://bulletin.miamioh.edu/farmer-business/",
  
  # FSB Admission
  "https://miamioh.edu/fsb/admission/highschool-students.html",
  "https://miamioh.edu/fsb/admission/transfer-students.html",
  "https://miamioh.edu/fsb/admission/current-students.html",
  
  # Forms and Approvals
  "https://miamioh.edu/fsb/student-resources/academic-advising/forms.html",
  
  # Academic Calendar
  "https://miamioh.edu/academic-programs/academic-calendar/2024-2025.html",
  
  # Other
  "https://miamioh.edu/regionals/student-resources/academic-advising/academic-status/gpa-calculator.html",
  "https://miamioh.edu/onestop/academic-records/access-order-transcripts.html",
  "https://miamioh.edu/onestop/registration/courses/credit-no-credit.html",
  "https://miamioh.edu/onestop/registration/courses/add-drop-course.html",
  "https://miamioh.formstack.com/forms/fsb_change_of_program",
  "https://miamioh.edu/onestop/transfer-credit/",
  "https://miamioh.edu/onestop/registration/holds-on-student-accounts.html",
  "https://miamioh.edu/onestop/registration/courses/final-exams/index.html",
  "https://miamioh.edu/onestop/registration/courses/banner-waitlist-ror.html",
  "https://miamioh.edu/onestop/registration/courses/excess-hours.html",
  "https://miamioh.edu/onestop/registration/courses/courses-at-another-miami-campus/index.html",
  "https://miamioh.edu/onestop/registration/courses/repeat-courses.html",
  "https://miamioh.edu/onestop/registration/courses/audit-class.html",
  "https://miamioh.edu/onestop/academic-records/president-deans-lists.html",
  "https://miamioh.edu/onestop/academic-records/graduation-diplomas/index.html",
  "https://miamioh.edu/onestop/academic-records/graduation-diplomas/degree-honors-distinction.html",
  "https://miamioh.edu/onestop/academic-records/graduation-diplomas/diplomas.html",
  "https://miamioh.edu/onestop/academic-records/academic-probation.html",
  "https://miamioh.edu/onestop/costs/oxford-undergrads/enrolled-2016-or-later/index.html",
  "https://miamioh.edu/integrity/policies-process",
  
  # Business Honors Program
  "https://miamioh.edu/fsb/academics/business-honors-program/index.html",
  "https://miamioh.edu/fsb/academics/business-honors-program/admission.html",
  "https://miamioh.edu/fsb/academics/business-honors-program/requirements.html",
  
  # Policy Library
  "https://miamioh.edu/fsb/academics/business-honors-program/requirements.html",
  
  # Courses of Instruction
  "https://bulletin.miamioh.edu/courses-instruction/acc/",
  "https://bulletin.miamioh.edu/courses-instruction/eco/",
  "https://bulletin.miamioh.edu/courses-instruction/esp/",
  "https://bulletin.miamioh.edu/courses-instruction/fin/",
  "https://bulletin.miamioh.edu/courses-instruction/isa/",
  "https://bulletin.miamioh.edu/courses-instruction/mgt/",
  "https://bulletin.miamioh.edu/courses-instruction/mkt/",
  "https://bulletin.miamioh.edu/courses-instruction/soc/",
  "https://bulletin.miamioh.edu/courses-instruction/psy/",
  "https://bulletin.miamioh.edu/courses-instruction/cse/",
  "https://bulletin.miamioh.edu/courses-instruction/bus/",
  "https://bulletin.miamioh.edu/courses-instruction/bls/",
  "https://bulletin.miamioh.edu/courses-instruction/sta/"
]


# Combining the URLs (focusing on the google doc URLs)
# ------------------------------------------------------------------------------
urls = urls_from_google_doc + absolute_urls


# Scraping the Contents of the URLs
# ------------------------------------------------------------------------------
loader = SeleniumURLLoader(urls = urls)
data = loader.load()


# Converting the Data into a String with XML Tags for each document (for Claude 3 models)
# ----------------------------------------------------------------------------------------

data_string = '<documents>'

index = 0
for document in data:
    data_string += f"<document index='{index}'>"
    data_string += "<source>"
    data_string += document.metadata['source']
    data_string += "</source>"
    data_string += "<document_content>"
    data_string += document.page_content
    data_string += "</document_content>"
    data_string += "</document>"
    index += 1

data_string += '</documents>'
  
# Number of words in the string and an approximate count of the number of tokens
num_words = np.char.count(data_string, ' ') + 1
approx_num_tokens = num_words * 1.30
print(f'Number of words in the string: {num_words}')
print(f'Approximate number of tokens: {approx_num_tokens}')


# data_string using markdown instead of XML tags (for future use with GPT-4o)
# ------------------------------------------------------------------------------
data_string_md = ''
index = 0
for document in data:
    data_string_md += f"# Document {index}\n"
    data_string_md += f"## Title: {document.metadata['title']}\n"
    data_string_md += f"## Source: {document.metadata['source']}\n"
    data_string_md += f"## Contents:\n"
    data_string_md += document.page_content
    data_string_md += '\n\n\n'
    index += 1

num_words = np.char.count(data_string_md, ' ') + 1
approx_num_tokens = num_words * 1.30
print(f'Number of words in the string: {num_words}')
print(f'Approximate number of tokens: {approx_num_tokens}')


# Saving the data
# ------------------------------------------------------------------------------
with open(os.path.join('data/website_urls.pkl'), 'wb') as f:
    pickle.dump(urls, f)

with open(os.path.join('data/website_data.pkl'), 'wb') as f:
    pickle.dump(data, f)
    
with open(os.path.join('data/website_data_string.pkl'), 'wb') as f:
    pickle.dump(data_string, f)
    
with open(os.path.join('data/website_data_string_md.pkl'), 'wb') as f:
    pickle.dump(data_string_md, f)
