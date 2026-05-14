# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.12-slim

WORKDIR /
ENV PYTHONPATH=/

COPY ./requirements.txt /requirements.txt
COPY ./LICENSE /LICENSE
COPY ./third_party_ip_notices.md /third_party_ip_notices.md
COPY ./nginx.conf /nginx.conf
COPY ./start_server.sh /start_server.sh

# Install nginx server
# https://gunicorn.org/deploy/?h=nginx#nginx-configuration
SHELL ["/bin/bash", "-c"]
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg2 ca-certificates lsb-release debian-archive-keyring libreadline8 libterm-readline-gnu-perl && \
    curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/mainline/debian `lsb_release -cs` nginx" | tee /etc/apt/sources.list.d/nginx.list && \
    echo -e "Package: *\nPin: origin nginx.org\nPin: release o=nginx\nPin-Priority: 900\n" | tee /etc/apt/preferences.d/99nginx && \
    apt-get install -y --no-install-recommends nginx && \
    rm /etc/nginx/nginx.conf && \
    apt-get purge -y curl

COPY ./src /src

RUN pip install --upgrade pip && \
  pip install --no-deps --ignore-requires-python --require-hashes -r requirements.txt && \
  python3 -m compileall -b / && \
  # Set permissions for all files and directories in /src to be readable by non-root user.
  find /src -type d -exec chmod 755 {} \; && \
  find /src -type f -exec chmod 644 {} \; && \
  # create non-root user and group to run the server as non-root user.
  groupadd -r NonRootUserGroup && useradd -r -m -g NonRootUserGroup NonRootUser && \
  # set permissions for nginx config to be readable by non-root user.
  chmod 644 /nginx.conf && \
  chmod 644 /LICENSE && \
  chmod 644 /third_party_ip_notices.md && \
  # set permissions for start_server.sh to be executable by non-root user.
  chmod 755 /start_server.sh && \
  chmod +x /start_server.sh

# Run unit tests on container.
RUN set -e && \
    python3 -m unittest discover -p "*_test.py" -s "/src" -t / && \
    set +e && \
    rm -rf /tmp/*
# Set container to run as non-root user.
USER NonRootUser
ENTRYPOINT ["/start_server.sh"]