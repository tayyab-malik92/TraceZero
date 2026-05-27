from jinja2 import Template

GDPR_TEMPLATE = Template("""
Subject: Right to Erasure Request under GDPR / CCPA – {{ user_name }}

To Whom It May Concern at {{ broker_name }},

I am writing to formally request the deletion of all personal data
associated with me that is held, processed, or distributed by your
organisation, in accordance with:

  • Article 17 of the GDPR ("Right to Erasure / Right to be Forgotten")
  • The California Consumer Privacy Act (CCPA), Section 1798.105

My identifying information is as follows:
  Full Name : {{ user_name }}
  City      : {{ user_city }}
  Date      : {{ date }}

Please confirm in writing within 30 days that all records matching
the above have been permanently deleted from your systems and are
no longer shared with third parties.

Failure to comply may result in a formal complaint being filed with
the relevant supervisory authority.

Yours sincerely,
{{ user_name }}
""")

def generate_removal_email(user_profile: dict, broker_name: str) -> str:
    from datetime import date
    return GDPR_TEMPLATE.render(
        user_name=user_profile.get("name", ""),
        user_city=user_profile.get("city", ""),
        broker_name=broker_name,
        date=date.today().strftime("%B %d, %Y"),
    )
