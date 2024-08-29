#pragma once

#include <stack>
#include <vector>
#include <deque>
#include <string>

std::string cuda_from_expression(const std::string &expression, const std::vector<std::string> &vars);

std::vector<std::string> parse_expression(const std::string &expression);

std::deque<std::string> tokenize(const std::string &expression);

bool is_operator(const std::string &token);

int precedence(const std::string &token);

bool right_assoc(const std::string &token);

std::string postfix_to_ff(const std::vector<std::string> &rpn);

std::string operator_to_function(const std::string &op, const std::string &L, const std::string &R);

std::string postfix_to_infix(const std::vector<std::string> &rpn);

