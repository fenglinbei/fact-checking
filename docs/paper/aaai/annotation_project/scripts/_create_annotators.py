# 在 label-studio shell (shell_plus) 中执行：创建两个标注账号并加入组织
from users.models import User
from organizations.models import Organization

org = Organization.objects.first()
print("RESULT org:", org.id if org else None, getattr(org, "title", ""))

accounts = [
    ("annotator1@annotation.local", "annotator1", "fc-annot-1-2026"),
    ("annotator2@annotation.local", "annotator2", "fc-annot-2-2026"),
]

for email, uname, pwd in accounts:
    u = User.objects.filter(email=email).first()
    if u is None:
        u = User.objects.create(email=email, username=uname, first_name=uname)
        print("RESULT created:", email)
    else:
        print("RESULT exists:", email)
    u.set_password(pwd)
    u.active_organization = org
    u.is_active = True
    u.save()
    # 加入组织
    try:
        org.add_user(u)
        print("RESULT added to org:", email)
    except Exception as e:
        print("RESULT add_user note:", email, repr(e))
    print("RESULT user:", email, "id=", u.id, "org=", u.active_organization_id)

# 汇总组织成员
from organizations.models import OrganizationMember
members = OrganizationMember.objects.filter(organization=org).values_list("user__email", flat=True)
print("RESULT org members:", list(members))
