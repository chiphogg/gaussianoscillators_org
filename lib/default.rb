# Copyright 2015 Google Inc. All Rights Reserved.
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
# limitations under the License.// Utility functions for animated plots.

# All files in the 'lib' directory will be loaded
# before nanoc starts compiling.

include Nanoc::Helpers::Blogging
include Nanoc::Helpers::LinkTo


class KnitrFilter < Nanoc::Filter
  identifier :knitr

  def run(content, params={})
    require 'tempfile'
    output_dir = 'output' + @item.identifier
    file = Tempfile.new('knitr_output')
    begin
      file.write(content)
      file.close
      command = ('Rscript -e \'library(knitr);' +
                 'dir.create(file.path(normalizePath("."), "' + output_dir +
                 '"), recursive=TRUE);' +
                 'dir <- normalizePath("' + output_dir + '");' +
                 'opts_knit$set(base.dir=dir, root.dir=normalizePath("."));' +
                 'opts_chunk$set(fig.path="", fig.cap="", fig.width=9);' +
                 'cat(knit(quiet=TRUE, output=NULL, input="' + file.path +
                 '"))\'')
      output_filename = `#{command}`
      result = IO.read(output_filename)
      File.delete(output_filename)
      return result
    ensure
      file.unlink
    end
  end
end


# Links to extra assets (javascript and CSS) requested in the yaml header.
def extra_asset_links
  lines = []
  if item.attributes.has_key?(:css)
    for stylesheet in item.attributes[:css]
      lines.push("<link href='/assets/css/#{stylesheet}.css'"+
                 " type='text/css' rel='stylesheet'>")
    end
  end
  if item.attributes.has_key?(:js)
    for script in item.attributes[:js]
      lines.push("<script type='text/javascript'" +
                 " src='/assets/js/#{script}.js'></script>")
    end
  end
  lines.join("\n")
end
