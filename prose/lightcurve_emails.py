names=['George','Djamel', 'Karim', 'Lyu', 'Tristan', 'FX', 'Amaury', 'George', 'Olga', 'Ph!l', 'Djamel', 'Karim']
emails=['georgina.dransfield@concordiastation.aq','mekarnia@oca.eu','karim.agabi@unice.fr','lyu.abe@unice.fr',
'guillotastep@gmail.com','schmider@oca.eu','a.triaud@bham.ac.uk','GXG831@student.bham.ac.uk','olga.suarez@oca.eu','philippe.bendjoya@oca.eu',
'djamel.mekarnia@concordiastation.aq', 'karim.agabi@concordiastation.aq']

import email, smtplib, ssl
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from prose import load
from prose.utils import make_email_table
from pathlib import Path
import os

def send_email(phot):

	obs=load(phot)

	sender_email='astep@concordiastation.aq' #ASTEP email address here
	user_name='astep' #ASTEP username here
	password='betapic' #Password in here

	for n, e in zip(names, emails):
	    subject = f"ASTEP Lightcurve of {obs.name}" #Email subject here
	    
	    #The body is plain text but you can format using f-strings. Example below. 
	    #body = f"Hi {n}, \n\nYou should get a .txt file with your name. \nPlease confirm receipt.\n\nTa,\nGeorge"
	    body=f"""\
	            <html>
	            <head>
	                <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,200;0,300;0,400;0,600;1,200;1,300;1,400&display=swap" rel="stylesheet">
	            </head>
	              <body>
	                <p style='font-family:"Source Sans Pro", Helvetica, sans-serif; font-size: 1.2rem'><b>Good morning {n},</b><br><br>
	                   Check out the reprocessed lightcurve for {obs.name}! <br><br>
	               
	                   Have a good'un,<br>
	                   <b>ASTEP @ Concordia Station</b><br>
	                   <br>

	                   <br>
	                   PS: Lemme know if you got this<br><br>
	                    <img style="width: 25rem" src="cid:lc"/>
	                    <img style="width: 25rem" src="cid:sys"/>
	                    <img style="width: 25rem" src="cid:st"/>
	                    <img style="width: 25rem" src="cid:tr"/>
	                </p>
	                {make_email_table(obs)}
	              </body>
	            </html>
	            """
	    
	    
	    # Create a multipart message and set headers
	    message = MIMEMultipart('related')
	    message["From"] = sender_email
	    message["To"] = e
	    message["Subject"] = subject

	    # Add body to email
	    #message.attach(MIMEText(body, "plain"))
	    message.attach(MIMEText(body, "html"))

	    #if os.path.exists(os.path.join(Path(phot).parent, 'transit_trend.png')):
	    #	lc_file=os.path.join(Path(phot).parent, 'transit_trend.png')
	    #else:
	    lc_file=os.path.join(Path(phot).parent, 'lightcurve.png')
	    sys_file=os.path.join(Path(phot).parent, 'systematics.png')
	    stars_file=os.path.join(Path(phot).parent, 'stars.png')
	    trend_file=os.path.join(Path(phot).parent, 'transit_trend.png')

	    with open(lc_file, 'rb') as fp:
	        img = MIMEImage(fp.read())
	        img.add_header("Content-ID", "<lc>")
	        message.attach(img)

	    with open(sys_file, 'rb') as fp:
	        img = MIMEImage(fp.read())
	        img.add_header("Content-ID", "<sys>")
	        message.attach(img)

	    with open(stars_file, 'rb') as fp:
	        img = MIMEImage(fp.read())
	        img.add_header("Content-ID", "<st>")
	        message.attach(img)

	    with open(trend_file, 'rb') as fp:
	        img = MIMEImage(fp.read())
	        img.add_header("Content-ID", "<tr>")
	        message.attach(img)
	    
	    # Add attachment to message and convert message to string
	    #message.attach(part)
	    text = message.as_string()

	    # Log in to server using secure context and send email
	    with smtplib.SMTP('192.168.8.3', 25) as server:
	        server.ehlo()
	        #server.starttls()
	        server.login(user_name, password)
	        server.sendmail(sender_email, e, text)
	        
	    time.sleep(1) #Since it's sending out loads of individual emails, we don't want the server thinking
	                    #you're a bot. 