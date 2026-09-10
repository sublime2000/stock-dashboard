# PythonAnywhere Deployment Guide

## Prerequisites
1. Sign up at [PythonAnywhere](https://www.pythonanywhere.com/) (Free tier available)
2. Your GitHub repo: `sublime2000/stock-dashboard`

## Step 1: Open a Bash Console
- Go to PythonAnywhere Dashboard
- Click **"Bash"** under **"New console"**

## Step 2: Clone Your Repository
```bash
git clone https://github.com/sublime2000/stock-dashboard.git
cd stock-dashboard
```

## Step 3: Create a Virtual Environment
```bash
mkvirtualenv --python=/usr/bin/python3.11 stock-dashboard-env
```

## Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 5: Configure the Web App
1. Go to **"Web"** tab in PythonAnywhere dashboard
2. Click **"Add a new web app"**
3. Select **"Manual configuration"** (NOT Flask)
4. Choose **Python 3.11**
5. Set the **Source code** directory: `/home/yourusername/stock-dashboard`

## Step 6: Edit WSGI Configuration
1. Click on the **"WSGI configuration file"** link (e.g., `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
2. Replace the contents with:

```python
import sys
import os

project_home = '/home/yourusername/stock-dashboard'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.environ['FLASK_ENV'] = 'production'

from app import app as application
```

3. Click **Save**

## Step 7: Configure Static Files (Optional)
Under **"Static files"** section:
- URL: `/static/`
- Directory: `/home/yourusername/stock-dashboard/static`

## Step 8: Reload the Web App
- Click the green **"Reload"** button
- Visit `https://yourusername.pythonanywhere.com`

## Important Notes

### Free Account Limitations
- App sleeps after inactivity (takes ~30s to wake up)
- Limited to 512 MB disk space
- Limited CPU time
- Outbound internet access restricted to whitelisted sites (yfinance may need whitelist)

### Whitelisting APIs (if needed)
If yfinance doesn't work on free tier, you may need to:
1. Upgrade to Hacker account ($5/month)
2. Or request API whitelisting

### Keeping the App Alive
- Enable **"Always-on task"** (requires paid account)
- Or use a cron job to ping your site periodically

## Updating the App
```bash
cd ~/stock-dashboard
git pull
# Restart the app
```

Then click **"Reload"** in the Web tab.

## Troubleshooting

### Check Error Logs
- Go to **Web** tab → **Error log** section

### Common Issues
1. **Import errors**: Make sure all packages installed in virtualenv
2. **Path issues**: Verify `project_home` path in wsgi.py
3. **Static files 404**: Check static files configuration

### Reload After Changes
Always click **"Reload"** after making any code changes!
