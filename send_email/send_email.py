from mlrun.execution import MLClientCtx
from typing import List

# Import the email modules we'll need
import smtplib
from email.message import EmailMessage
import os

# For guessing MIME type based on file name extension
import mimetypes


def send_email(
    context: MLClientCtx,
    sender: str,
    to: str,
    subject: str,
    content: str = "",
    server_addr: str = None,
    attachments: List[str] = [],
) -> None:
    """Send an email.
    :param sender: Sender email address
    :param context: The function context
    :param to: Email address of mail recipient
    :param subject: Email subject
    :param content: Optional mail text
    :param server_addr: Address of SMTP server to use. Use format <addr>:<port>
    :param attachments: List of attachments to add.
    """

    # Validate inputs
    email_user = context.get_secret("SMTP_USER")
    email_pass = context.get_secret("SMTP_PASSWORD")
    if email_user is None or email_pass is None:
        context.logger.error("Missing sender email or password - cannot send email.")
        return

    if server_addr is None:
        context.logger.error("Server not specified - cannot send email.")
        return

    msg = EmailMessage()
    msg["From"] = sender
    msg["Subject"] = subject
    msg["To"] = to
    msg.set_content(content)

    for filename in attachments:
        context.logger.info(f"Looking at attachment: {filename}")
        if not os.path.isfile(filename):
            context.logger.warning(f"Filename does not exist {filename}")
            continue
        # Guess the content type based on the file's extension.  Encoding
        # will be ignored, although we should check for simple things like
        # gzip'd or compressed files.
        ctype, encoding = mimetypes.guess_type(filename)
        if ctype is None or encoding is not None:
            # No guess could be made, or the file is encoded (compressed), so
            # use a generic bag-of-bits type.
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(filename, "rb") as fp:
            msg.add_attachment(
                fp.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(filename),
            )
            context.logger.info(
                f"Added attachment: Filename: {filename}, of mimetype: {maintype}, {subtype}"
            )

    try:
        s = smtplib.SMTP(host=server_addr)
        s.starttls()
        s.login(email_user, email_pass)
        s.send_message(msg)
        context.logger.info("Email sent successfully.")
    except smtplib.SMTPException as exp:
        context.logger.error(f"SMTP exception caught in SMTP code: {exp}")
    except ConnectionError as ce:
        context.logger.error(f"Connection error caught in SMTP code: {ce}")
