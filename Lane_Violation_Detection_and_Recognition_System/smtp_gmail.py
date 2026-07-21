import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# from email.mime.application import MIMEApplication

import os
from dotenv import load_dotenv

try:
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

except (FileNotFoundError, KeyError) as e:
    print(f"ERROR: Credentials not found or incomplete: {e}")


def send_email_with_smtp_gmail(recipient, subject, body, image_path):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, "rb") as f:
        img = MIMEImage(f.read())
    img.add_header('Content-Disposition', 'attachment', filename=str(image_path))
    msg.attach(img)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        server.close()
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
