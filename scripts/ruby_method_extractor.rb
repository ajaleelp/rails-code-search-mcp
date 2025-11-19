#!/usr/bin/env ruby
require 'json'
require 'ripper'

path = ARGV[0]
abort 'usage: ruby_method_extractor.rb <file.rb>' unless path && File.file?(path)

code = File.read(path)
lines = code.lines
tokens = Ripper.lex(code)

def collect_name(tokens, start_index)
  name_parts = []
  i = start_index
  while i < tokens.length
    _pos, type, text, _state = tokens[i]

    break if [:on_nl, :on_ignored_nl, :on_comment].include?(type) && name_parts.any?
    break if type == :on_kw && %w[do end if unless while until].include?(text)
    if type == :on_sp
      if name_parts.any?
        break
      else
        i += 1
        next
      end
    end

    stop_token = type == :on_op && ['<', '(', ')', ';', '=', ',', '['].include?(text)
    break if stop_token

    allowed = [:on_ident, :on_const, :on_cvar, :on_ivar, :on_gvar]
    if allowed.include?(type)
      name_parts << text
      i += 1
      next
    end

    if type == :on_kw && text == 'self'
      name_parts << text
      i += 1
      next
    end

    if type == :on_op && ['::', '.', '@', '&', '[]', '[]='].include?(text)
      name_parts << text
      i += 1
      next
    end

    break
  end

  [name_parts.join.strip, i]
end

block_stack = []
class_stack = []
methods = []

block_keywords = %w[do begin case for while until if unless]

tokens.each_with_index do |token, idx|
  (pos, type, text, _state) = token
  line = pos[0]

  next unless type == :on_kw

  case text
  when 'class', 'module'
    name, _ = collect_name(tokens, idx + 1)
    class_stack << name unless name.empty?
    block_stack << { type: text, name: name, start_line: line }
  when 'def'
    name, _ = collect_name(tokens, idx + 1)
    qualified_class = class_stack.compact.join('::') if class_stack.any?
    block_stack << { type: 'def', name: name, start_line: line, class_name: qualified_class }
  when 'end'
    entry = block_stack.pop
    next unless entry

    case entry[:type]
    when 'def'
      methods << entry.merge(end_line: line)
    when 'class', 'module'
      class_stack.pop
    end
  when *block_keywords
    block_stack << { type: 'block' }
  end
end

methods.each do |method_info|
  start_line = method_info[:start_line]
  end_line = method_info[:end_line]
  snippet = lines[(start_line - 1)..(end_line - 1)]&.join || ''
  method_info[:path] = path
  method_info[:text] = snippet
method_info[:class_name] = method_info.delete(:class_name)
method_info[:method_name] = method_info.delete(:name) || method_info.delete(:method)
end

puts JSON.generate(methods)
