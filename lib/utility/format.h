//
// Created by richard on 17.02.21.
//

#ifndef FORMAT_H
#define FORMAT_H

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline const char *getFormat() {
  throw std::runtime_error("Type formater not implemented yet");
}

template<>
inline const char *getFormat<int>() {
  return "%d";
}

template<>
inline const char *getFormat<float>() {
  return "%f";
}

template<>
inline const char *getFormat<double>() {
  return getFormat<float>();
}

#endif //FORMAT_H
